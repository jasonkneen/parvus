import numpy as np
import json
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import torch


class ParvusCompressor:
    def __init__(self):
        self.compressed_data = None
        self.decompressor = None
        self.faiss_index = None
        
        # Check if CUDA is available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU Device: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # Initialize the model with GPU support if available
        self.text_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        if torch.cuda.is_available():
            self.text_embedding_model = self.text_embedding_model.to(self.device)
            print("SentenceTransformer model loaded on GPU")
        else:
            print("Running on CPU - for better performance, consider using a CUDA-enabled GPU")
        
        self.original_messages = []
        self.compression_ready = False

    def load_data_from_json(self, file_path):
        """Load data from a JSON file."""
        print("Loading data from JSON...")
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Handle messages format
        if isinstance(data, dict) and 'messages' in data:
            messages = data['messages']
            self.original_messages = messages
            texts = [msg['content'] for msg in messages]
        elif isinstance(data, list):
            self.original_messages = data
            texts = [msg.get('content', msg) for msg in data]
        else:
            raise ValueError("JSON must contain either a 'messages' array or be an array of messages")

        # Generate embeddings with GPU acceleration if available
        print(f"Generating embeddings for {len(texts)} messages...")
        if torch.cuda.is_available():
            embeddings = self.text_embedding_model.encode(
                texts,
                batch_size=32,  # Adjust based on your GPU memory
                convert_to_numpy=True,
                device=self.device
            )
        else:
            embeddings = self.text_embedding_model.encode(texts)
        
        return embeddings

    def build_faiss_index(self):
        """Build the FAISS index from compressed data with GPU support if available."""
        print("Building FAISS index...")
        if self.compressed_data is None or self.compressed_data.size == 0:
            raise ValueError("Cannot build FAISS index: Compressed data is empty.")
        
        # Create index based on dimension
        dimension = self.compressed_data.shape[1]
        
        if torch.cuda.is_available():
            # Use GPU index
            res = faiss.StandardGpuResources()
            config = faiss.GpuIndexFlatConfig()
            config.device = 0  # GPU device to use
            
            cpu_index = faiss.IndexFlatL2(dimension)
            self.faiss_index = faiss.GpuIndexFlatL2(res, dimension, config)
            self.faiss_index.add(self.compressed_data.astype(np.float32))
            print("Using GPU-accelerated FAISS index")
        else:
            # Use CPU index
            self.faiss_index = faiss.IndexFlatL2(dimension)
            self.faiss_index.add(self.compressed_data.astype(np.float32))
            print("Using CPU FAISS index")
            
        print(f"FAISS index contains {self.faiss_index.ntotal} entries")

    def compress(self, data):
        """Compress the input data using TruncatedSVD."""
        print(f"Compressing dataset with {len(self.original_messages)} messages...")
        
        # Convert to numpy array if not already
        data = np.array(data)
        
        # Calculate compression parameters
        n_components = min(data.shape[0] // 2, data.shape[1])
        n_components = max(2, n_components)  # Ensure at least 2 components
        
        # Initialize and fit TruncatedSVD
        self.decompressor = TruncatedSVD(n_components=n_components)
        self.compressed_data = self.decompressor.fit_transform(data)
        
        # Build FAISS index
        self.build_faiss_index()
        
        # Calculate compression metrics
        original_size = data.nbytes
        compressed_size = self.compressed_data.nbytes
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
        
        print(f"Original size: {original_size} bytes")
        print(f"Compressed size: {compressed_size} bytes")
        print(f"Compression ratio: {compression_ratio:.2f}")
        print(f"Number of messages stored: {len(self.original_messages)}")
        
        if compression_ratio < 1:
            print("Warning: Compression was not effective. Consider adjusting parameters.")
        
        self.compression_ready = True
        return self.compressed_data

    def query(self, query_input, top_k=5, is_text_query=True):
        """Query the dataset using either a text query or a numeric vector."""
        if not self.compression_ready:
            raise RuntimeError("The compression system is not ready. Please upload and compress a dataset first.")

        print("Querying data...")
        
        # Encode query input
        if is_text_query:
            query_vector = self.text_embedding_model.encode(query_input)
        else:   
            query_vector = query_input

        # Ensure query_vector is valid
        if query_vector is None or (isinstance(query_vector, np.ndarray) and query_vector.size == 0):
            raise ValueError("Query vector is empty or invalid.")

        # Transform query vector to match compressed data dimensions
        query_vector_reduced = self.decompressor.transform([query_vector])
        
        # Ensure we don't request more items than we have
        top_k = min(top_k, len(self.original_messages))
        if top_k == 0:
            raise ValueError("No messages available to search through")
            
        distances, indices = self.faiss_index.search(query_vector_reduced, top_k)

        print(f"Found {len(indices[0])} results")
        print(f"Distances: {distances}")
        print(f"Indices: {indices}")

        # Ensure indices are valid
        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.original_messages):
                results.append({
                    "role": self.original_messages[int(idx)].get('role', 'Unknown'),
                    "message": self.original_messages[int(idx)].get('content', self.original_messages[int(idx)])
                })

        if not results:
            print("No valid results found")
            return [], []

        return results, distances[0][:len(results)]

    def save_state(self, data_file, index_file):
        """Save the compressed state to files."""
        if not self.compression_ready:
            raise RuntimeError("No compressed data to save.")
        
        # Save compressed data and decompressor
        with open(data_file, 'wb') as f:
            pickle.dump({
                'compressed_data': self.compressed_data,
                'decompressor': self.decompressor,
                'original_messages': self.original_messages
            }, f)
        
        # Save FAISS index
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, index_file)

    def load_state(self, data_file, index_file):
        """Load the compressed state from files."""
        # Load compressed data and decompressor
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
            self.compressed_data = data['compressed_data']
            self.decompressor = data['decompressor']
            self.original_messages = data['original_messages']
        
        # Load FAISS index
        self.faiss_index = faiss.read_index(index_file)
        self.compression_ready = True

    def get_compression_details(self):
        """Get details about the compression."""
        if not self.compression_ready:
            return None
        
        return {
            "Original Dimension": self.decompressor.components_.shape[1],
            "Compressed Dimension": self.compressed_data.shape[1],
            "Number of Items": len(self.original_messages),
            "Compression Ratio": self.decompressor.components_.shape[1] / self.compressed_data.shape[1]
        }


# Test Example
if __name__ == "__main__":
    # Initialize the compression system
    compression_system = ParvusCompressor()
    
    # Test data
    test_data = [
        {"role": "user", "message": "Hello, how are you?"},
        {"role": "assistant", "message": "I'm doing well, thank you!"},
        {"role": "user", "message": "What's the weather like today?"},
        {"role": "assistant", "message": "I don't have access to current weather information."},
    ]
    
    # Save test data to JSON
    with open('test_data.json', 'w') as f:
        json.dump(test_data, f)
    
    try:
        # Load and compress data
        embeddings = compression_system.load_data_from_json('test_data.json')
        compression_system.compress(embeddings)
        
        # Test querying
        print("\nTesting query functionality:")
        results, distances = compression_system.query("How's the weather?", top_k=2)
        
        # Save state
        compression_system.save_state("test_compressed.pkl", "test_index.bin")
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {str(e)}")
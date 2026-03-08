#!/usr/bin/env python3
"""
ChromaDB Embedding Pipeline for NASA Space Mission Data - Text Files Only

This script reads parsed text data from various NASA space mission folders
and creates a permanent ChromaDB collection with OpenAI embeddings
for RAG applications.
Optimized to process only text files to avoid duplication with JSON versions.

Supported data sources:
- Apollo 11 extracted data (text files only)
- Apollo 13 extracted data (text files only)
- Apollo 11 Textract extracted data (text files only)
- Challenger transcribed audio data (text files only)
"""

import argparse
import hashlib
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("chroma_embedding_text_only.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class ChromaEmbeddingPipelineTextOnly:
    """Pipeline for creating ChromaDB collections with OpenAI embeddings."""

    def __init__(
        self,
        openai_api_key: str,
        chroma_persist_directory: str = "./chroma_db",
        collection_name: str = "nasa_space_missions_text",
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 800,
        chunk_overlap: int = 120,
    ):
        """
        Initialize the embedding pipeline

        Args:
            openai_api_key: OpenAI API key
            chroma_persist_directory: Directory to persist ChromaDB
            collection_name: Name of the ChromaDB collection
            embedding_model: OpenAI embedding model to use
            chunk_size: Maximum size of text chunks
            chunk_overlap: Overlap between chunks
        """
        # Store configuration parameters
        self.openai_api_key = openai_api_key
        self.chroma_persist_directory = chroma_persist_directory
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=chroma_persist_directory)

        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=OpenAIEmbeddingFunction(
                api_key=openai_api_key, model_name=embedding_model
            ),
        )

        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=openai_api_key)

    def chunk_text(
        self, text: str, metadata: Dict[str, Any]
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Split text into token-safe chunks with metadata.
        """

        import re

        import tiktoken

        tokenizer = tiktoken.encoding_for_model(self.embedding_model)

        # Safe limits for OpenAI embeddings
        MAX_MODEL_TOKENS = 8192
        SAFETY_MARGIN = 200

        max_tokens = min(self.chunk_size, MAX_MODEL_TOKENS - SAFETY_MARGIN)
        overlap_tokens = self.chunk_overlap

        # Better splitting for transcripts + documents
        sentences = re.split(r"(?<=[.!?])\s+|\n+", text)

        chunks = []
        current_chunk = []
        current_tokens = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            tokens = tokenizer.encode(sentence)
            sentence_tokens = len(tokens)

            # If one sentence is too large → split by words
            if sentence_tokens > max_tokens:
                words = sentence.split()
                buffer = []

                for word in words:
                    if not buffer:
                        word_token_count = len(tokenizer.encode(word))
                        if word_token_count > max_tokens:
                            word_tokens = tokenizer.encode(word)
                            for start in range(
                                0,
                                len(word_tokens),
                                max_tokens,
                            ):
                                chunk_text = tokenizer.decode(
                                    word_tokens[start : start + max_tokens]
                                )

                                chunk_metadata = metadata.copy()
                                chunk_metadata["chunk_index"] = len(chunks) + 1
                                chunks.append((chunk_text, chunk_metadata))
                            continue

                        buffer = [word]
                        continue

                    candidate_buffer = buffer + [word]
                    token_count = len(
                        tokenizer.encode(" ".join(candidate_buffer))
                    )

                    if token_count > max_tokens:
                        chunk_text = " ".join(buffer)

                        chunk_metadata = metadata.copy()
                        chunk_metadata["chunk_index"] = len(chunks) + 1

                        chunks.append((chunk_text, chunk_metadata))

                        word_token_count = len(tokenizer.encode(word))
                        if word_token_count > max_tokens:
                            word_tokens = tokenizer.encode(word)
                            for start in range(
                                0,
                                len(word_tokens),
                                max_tokens,
                            ):
                                chunk_text = tokenizer.decode(
                                    word_tokens[start : start + max_tokens]
                                )

                                chunk_metadata = metadata.copy()
                                chunk_metadata["chunk_index"] = len(chunks) + 1
                                chunks.append((chunk_text, chunk_metadata))
                            buffer = []
                        else:
                            buffer = [word]
                    else:
                        buffer = candidate_buffer

                if buffer:
                    sentences.insert(0, " ".join(buffer))

                continue

            # If adding sentence exceeds chunk size → finalize chunk
            if current_tokens + sentence_tokens > max_tokens:
                chunk_text = " ".join(current_chunk)

                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_index"] = len(chunks) + 1

                chunks.append((chunk_text, chunk_metadata))

                # Token-based overlap
                overlap_chunk = []
                overlap_count = 0

                for s in reversed(current_chunk):
                    s_tokens = len(tokenizer.encode(s))

                    if overlap_count + s_tokens > overlap_tokens:
                        break

                    overlap_chunk.insert(0, s)
                    overlap_count += s_tokens

                current_chunk = overlap_chunk
                current_tokens = overlap_count

            current_chunk.append(sentence)
            current_tokens += sentence_tokens

        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)

            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = len(chunks) + 1

            chunks.append((chunk_text, chunk_metadata))

        return chunks

    def check_document_exists(self, doc_id: str) -> bool:
        """
        Check if a document with the given ID already exists in the collection

        Args:
            doc_id: Document ID to check

        Returns:
            True if document exists, False otherwise
        """
        try:
            # Query the collection for the document ID
            results = self.collection.get(ids=[doc_id])

            # If the document is found, return True
            return len(results["ids"]) > 0
        except Exception as e:
            logger.error(
                f"Error checking document existence for ID {doc_id}: {e}"
            )
            return False

    def update_document(
        self,
        doc_id: str,
        text: str,
        metadata: Dict[str, Any],
    ) -> bool:
        """
        Update an existing document in the collection

        Args:
            doc_id: Document ID to update
            text: New text content
            metadata: New metadata

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get new embedding
            embedding = self.get_embedding(text)

            # Update the document
            self.collection.update(
                ids=[doc_id],
                documents=[text],
                metadatas=[metadata],
                embeddings=[embedding],
            )
            logger.debug(f"Updated document: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating document {doc_id}: {e}")
            return False

    def delete_documents_by_source(self, source_pattern: str) -> int:
        """
        Delete all documents from a specific source.

        Args:
            source_pattern: Pattern to match source names

        Returns:
            Number of documents deleted
        """
        try:
            # Get all documents
            all_docs = self.collection.get()

            # Find documents matching the source pattern
            ids_to_delete = []
            for i, metadata in enumerate(all_docs["metadatas"]):
                if source_pattern in metadata.get("source", ""):
                    ids_to_delete.append(all_docs["ids"][i])

            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                logger.info(
                    "Deleted %s documents matching source pattern: %s",
                    len(ids_to_delete),
                    source_pattern,
                )
                return len(ids_to_delete)
            else:
                logger.info(
                    "No documents found matching source pattern: %s",
                    source_pattern,
                )
                return 0

        except Exception as e:
            logger.error(f"Error deleting documents by source: {e}")
            return 0

    def get_file_documents(self, file_path: Path) -> List[str]:
        """
        Get all document IDs for a specific file

        Args:
            file_path: Path to the file

        Returns:
            List of document IDs for the file
        """
        try:
            source = file_path.stem
            mission = self.extract_mission_from_path(file_path)

            # Get all documents
            all_docs = self.collection.get()

            # Find documents from this file
            file_doc_ids = []
            for i, metadata in enumerate(all_docs["metadatas"]):
                if (
                    metadata.get("source") == source
                    and metadata.get("mission") == mission
                ):
                    file_doc_ids.append(all_docs["ids"][i])

            return file_doc_ids

        except Exception as e:
            logger.error(f"Error getting file documents: {e}")
            return []

    def get_embedding(self, text: str) -> List[float]:
        """
        Get OpenAI embedding for text

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        try:
            # Call OpenAI API to get embeddings
            response = self.openai_client.embeddings.create(
                input=text, model=self.embedding_model
            )

            # Extract the embedding vector from the response
            embedding = response.data[0].embedding
            return embedding

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return []

    def generate_document_id(
        self,
        file_path: Path,
        metadata: Dict[str, Any],
    ) -> str:
        """
        Generate stable document ID based on file path and chunk position
        This allows for document updates without changing IDs

        Args:
            file_path: Path to the file
            metadata: Metadata dictionary containing chunk index

        Returns:
            A unique document ID string
        """
        # Extract mission and source from metadata
        mission = metadata.get("mission", "unknown")
        source = metadata.get("source", file_path.stem)
        chunk_index = metadata.get("chunk_index", 0)
        chunk_index_padded = str(chunk_index).zfill(4)

        # Create a consistent ID format: mission_source_chunk_0001
        document_id = f"{mission}_{source}_chunk_{chunk_index_padded}"

        # Hash the ID to ensure uniqueness if needed
        hashed_id = hashlib.md5(document_id.encode()).hexdigest()

        return hashed_id

    def process_text_file(
        self,
        file_path: Path,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Process plain text files with enhanced metadata extraction

        Args:
            file_path: Path to text file

        Returns:
            List of (text, metadata) tuples
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            if not content.strip():
                return []

            # Enhanced metadata extraction
            document_category = self.extract_document_category_from_filename(
                file_path.name
            )
            metadata = {
                "source": file_path.stem,
                "file_path": str(file_path),
                "file_type": "text",
                "content_type": "full_text",
                "mission": self.extract_mission_from_path(file_path),
                "data_type": self.extract_data_type_from_path(file_path),
                "document_category": document_category,
                "file_size": len(content),
                "processed_timestamp": datetime.now().isoformat(),
            }

            return self.chunk_text(content, metadata)

        except Exception as e:
            logger.error(f"Error processing text file {file_path}: {e}")
            return []

    def extract_mission_from_path(self, file_path: Path) -> str:
        """Extract mission name from file path"""
        path_str = str(file_path).lower()
        if "apollo11" in path_str or "apollo_11" in path_str:
            return "apollo_11"
        elif "apollo13" in path_str or "apollo_13" in path_str:
            return "apollo_13"
        elif "challenger" in path_str:
            return "challenger"
        else:
            return "unknown"

    def extract_data_type_from_path(self, file_path: Path) -> str:
        """Extract data type from file path"""
        path_str = str(file_path).lower()
        if "transcript" in path_str:
            return "transcript"
        elif "textract" in path_str:
            return "textract_extracted"
        elif "audio" in path_str:
            return "audio_transcript"
        elif "flight_plan" in path_str:
            return "flight_plan"
        else:
            return "document"

    def extract_document_category_from_filename(self, filename: str) -> str:
        """Extract document category from filename for better organization"""
        filename_lower = filename.lower()

        # Apollo transcript types
        if "pao" in filename_lower:
            return "public_affairs_officer"
        elif "cm" in filename_lower:
            return "command_module"
        elif "tec" in filename_lower:
            return "technical"
        elif "flight_plan" in filename_lower:
            return "flight_plan"

        # Challenger audio segments
        elif "mission_audio" in filename_lower:
            return "mission_audio"

        # NASA archive documents
        elif "ntrs" in filename_lower:
            return "nasa_archive"
        elif "19900066485" in filename_lower:
            return "technical_report"
        elif "19710015566" in filename_lower:
            return "mission_report"

        # General categories
        elif "full_text" in filename_lower:
            return "complete_document"
        else:
            return "general_document"

    def scan_text_files_only(self, base_path: str) -> List[Path]:
        """
        Scan data directories for text files only (avoiding JSON duplicates)

        Args:
            base_path: Base directory path

        Returns:
            List of text file paths to process
        """
        base_path = Path(base_path)
        files_to_process = []

        # Define directories to scan
        data_dirs = ["apollo11", "apollo13", "challenger"]

        for data_dir in data_dirs:
            dir_path = base_path / data_dir
            if dir_path.exists():
                logger.info(f"Scanning directory: {dir_path}")

                # Find only text files
                text_files = list(dir_path.glob("**/*.txt"))
                files_to_process.extend(text_files)
                logger.info(
                    "Found %s text files in %s",
                    len(text_files),
                    data_dir,
                )

        # Filter out unwanted files
        filtered_files = []
        for file_path in files_to_process:
            # Skip system files and summaries
            if (
                file_path.name.startswith(".")
                or "summary" in file_path.name.lower()
                or file_path.suffix.lower() != ".txt"
            ):
                continue
            filtered_files.append(file_path)

        logger.info(f"Total text files to process: {len(filtered_files)}")

        # Log file breakdown by mission
        mission_counts = {}
        for file_path in filtered_files:
            mission = self.extract_mission_from_path(file_path)
            mission_counts[mission] = mission_counts.get(mission, 0) + 1

        logger.info("Files by mission:")
        for mission, count in mission_counts.items():
            logger.info(f"  {mission}: {count} files")

        return filtered_files

    def add_documents_to_collection(
        self,
        documents: List[Tuple[str, Dict[str, Any]]],
        file_path: Path,
        batch_size: int = 50,
        update_mode: str = "skip",
    ) -> Dict[str, int]:
        """
        Add documents to ChromaDB collection in batches with update handling

        Args:
            documents: List of (text, metadata) tuples
            file_path: Path to the source file
            batch_size: Number of documents to process in each batch
            update_mode: How to handle existing documents:
                        'skip' - skip existing documents
                        'update' - update existing documents
                        'replace' - delete all existing documents
                        from file and re-add

        Returns:
            Dictionary with counts of added, updated, and skipped documents
        """
        if not documents:
            return {"added": 0, "updated": 0, "skipped": 0}

        stats = {"added": 0, "updated": 0, "skipped": 0}

        # Handle replace mode by deleting existing documents
        if update_mode == "replace":
            existing_doc_ids = self.get_file_documents(file_path)
            if existing_doc_ids:
                self.collection.delete(ids=existing_doc_ids)

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            ids, texts, metadatas, embeddings = [], [], [], []

            for text, metadata in batch:
                doc_id = self.generate_document_id(file_path, metadata)
                exists = self.check_document_exists(doc_id)

                if exists:
                    if update_mode == "skip":
                        stats["skipped"] += 1
                        continue
                    elif update_mode == "update":
                        self.update_document(doc_id, text, metadata)
                        stats["updated"] += 1
                        continue

                # Generate embedding and prepare for addition
                embedding = self.get_embedding(text)
                ids.append(doc_id)
                texts.append(text)
                metadatas.append(metadata)
                embeddings.append(embedding)

                stats["added"] += 1

            # Add batch to collection
            if ids:
                self.collection.add(
                    ids=ids,
                    documents=texts,
                    metadatas=metadatas,
                    embeddings=embeddings,
                )

        return stats

    def process_all_text_data(
        self,
        base_path: str,
        update_mode: str = "skip",
        batch_size: int = 50,
    ) -> Dict[str, int]:
        """
        Process all text files and add to ChromaDB

        Args:
            base_path: Base directory containing data folders
            update_mode: How to handle existing documents:
                        'skip' - skip existing documents (default)
                        'update' - update existing documents
                        'replace' - delete all existing documents
                        from file and re-add

        Returns:
            Statistics about processed files
        """
        stats = {
            "files_processed": 0,
            "documents_added": 0,
            "documents_updated": 0,
            "documents_skipped": 0,
            "errors": 0,
            "total_chunks": 0,
            "missions": {},
        }

        try:
            files_to_process = self.scan_text_files_only(base_path)

            for file_path in files_to_process:
                try:
                    # Process the file into chunks
                    chunks = self.process_text_file(file_path)

                    # Add chunks to the collection
                    file_stats = self.add_documents_to_collection(
                        chunks,
                        file_path,
                        batch_size=batch_size,
                        update_mode=update_mode,
                    )

                    # Update statistics
                    stats["files_processed"] += 1
                    stats["documents_added"] += file_stats["added"]
                    stats["documents_updated"] += file_stats["updated"]
                    stats["documents_skipped"] += file_stats["skipped"]
                    stats["total_chunks"] += len(chunks)

                    # Update mission stats
                    mission = self.extract_mission_from_path(file_path)
                    if mission not in stats["missions"]:
                        stats["missions"][mission] = {
                            "files": 0,
                            "chunks": 0,
                            "added": 0,
                            "updated": 0,
                            "skipped": 0,
                        }
                    stats["missions"][mission]["files"] += 1
                    stats["missions"][mission]["chunks"] += len(chunks)
                    stats["missions"][mission]["added"] += file_stats["added"]
                    stats["missions"][mission]["updated"] += file_stats[
                        "updated"
                    ]
                    stats["missions"][mission]["skipped"] += file_stats[
                        "skipped"
                    ]

                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
                    stats["errors"] += 1

        except Exception as e:
            logger.error(f"Error scanning text files: {e}")
            stats["errors"] += 1

        return stats

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the ChromaDB collection

        Returns:
            Dictionary containing collection name, document count, and metadata
        """
        try:
            # Get all documents in the collection
            all_docs = self.collection.get()

            # Extract collection information
            collection_info = {
                "collection_name": self.collection_name,
                "document_count": len(all_docs["ids"]),
                "metadata": {
                    "missions": {},
                    "data_types": {},
                    "document_categories": {},
                    "file_types": {},
                },
            }

            # Analyze metadata
            for metadata in all_docs["metadatas"]:
                mission = metadata.get("mission", "unknown")
                data_type = metadata.get("data_type", "unknown")
                doc_category = metadata.get("document_category", "unknown")
                file_type = metadata.get("file_type", "unknown")

                # Count by mission
                collection_info["metadata"]["missions"][mission] = (
                    collection_info["metadata"]["missions"].get(mission, 0) + 1
                )

                # Count by data type
                collection_info["metadata"]["data_types"][data_type] = (
                    collection_info["metadata"]["data_types"].get(data_type, 0)
                    + 1
                )

                # Count by document category
                doc_categories = collection_info["metadata"][
                    "document_categories"
                ]
                doc_categories[doc_category] = (
                    collection_info["metadata"]["document_categories"].get(
                        doc_category, 0
                    )
                    + 1
                )

                # Count by file type
                collection_info["metadata"]["file_types"][file_type] = (
                    collection_info["metadata"]["file_types"].get(file_type, 0)
                    + 1
                )

            return collection_info

        except Exception as e:
            logger.error(f"Error retrieving collection info: {e}")
            return {"error": str(e)}

    def get_collection_stats(self) -> Dict[str, Any]:
        """Compatibility wrapper for collection statistics output."""
        return self.get_collection_info()

    def query_collection(
        self,
        query: str,
        n_results: int = 3,
        mission_filter: str = "",
    ) -> Dict[str, Any]:
        """Run similarity search with optional mission metadata filter."""
        try:
            where_filter = None
            if mission_filter and mission_filter.lower() != "all":
                where_filter = {"mission": mission_filter}

            return self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_filter,
            )
        except Exception as e:
            logger.error(f"Error running collection query: {e}")
            return {"error": str(e)}


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="ChromaDB Embedding Pipeline for NASA Data"
    )
    parser.add_argument(
        "--data-path",
        default=".",
        help="Path to data directories",
    )
    parser.add_argument(
        "--openai-key",
        default=os.getenv("OPENAI_KEY"),
        help="OpenAI API key",
    )
    parser.add_argument(
        "--chroma-dir",
        default="./chroma_db_openai",
        help="ChromaDB persist directory",
    )
    parser.add_argument(
        "--collection-name",
        default="nasa_space_missions_text",
        help="Collection name",
    )
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
        help="OpenAI embedding model",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Text chunk size",
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=120, help="Chunk overlap size"
    )
    parser.add_argument(
        "--batch-size", type=int, default=50, help="Batch size for processing"
    )
    parser.add_argument(
        "--update-mode",
        choices=["skip", "update", "replace"],
        default="skip",
        help="How to handle existing documents: skip, update, or replace",
    )
    parser.add_argument("--test-query", help="Test query after processing")
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only show collection statistics",
    )
    parser.add_argument(
        "--delete-source",
        help="Delete all documents from a specific source pattern",
    )

    args = parser.parse_args()

    # Initialize pipeline
    logger.info("Initializing ChromaDB Embedding Pipeline...")
    pipeline = ChromaEmbeddingPipelineTextOnly(
        openai_api_key=args.openai_key,
        chroma_persist_directory=args.chroma_dir,
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    # Handle delete source operation
    if args.delete_source:
        deleted_count = pipeline.delete_documents_by_source(args.delete_source)
        logger.info(
            "Deleted %s documents matching source pattern: %s",
            deleted_count,
            args.delete_source,
        )
        return

    # If stats only, show collection statistics and exit
    if args.stats_only:
        logger.info("Collection Statistics:")
        stats = pipeline.get_collection_stats()
        for key, value in stats.items():
            logger.info(f"{key}: {value}")
        return

    # Process all data
    logger.info(
        "Starting text data processing with update mode: %s",
        args.update_mode,
    )
    start_time = time.time()

    stats = pipeline.process_all_text_data(
        args.data_path,
        update_mode=args.update_mode,
        batch_size=args.batch_size,
    )

    end_time = time.time()
    processing_time = end_time - start_time

    # Print results
    logger.info("=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Files processed: {stats['files_processed']}")
    logger.info(f"Total chunks created: {stats['total_chunks']}")
    logger.info(f"Documents added to collection: {stats['documents_added']}")
    logger.info(
        f"Documents updated in collection: {stats['documents_updated']}"
    )
    logger.info(
        f"Documents skipped (already exist): {stats['documents_skipped']}"
    )
    logger.info(f"Errors: {stats['errors']}")
    logger.info("Processing time: {:.2f} seconds".format(processing_time))

    # Mission breakdown
    logger.info("\nMission breakdown:")
    for mission, mission_stats in stats["missions"].items():
        logger.info(
            "  %s: %s files, %s chunks",
            mission,
            mission_stats["files"],
            mission_stats["chunks"],
        )
        logger.info(
            "    Added: %s, Updated: %s, Skipped: %s",
            mission_stats["added"],
            mission_stats["updated"],
            mission_stats["skipped"],
        )

    # Collection info
    collection_info = pipeline.get_collection_info()
    logger.info(
        f"\nCollection: {collection_info.get('collection_name', 'N/A')}"
    )
    logger.info(
        "Total documents in collection: %s",
        collection_info.get("document_count", "N/A"),
    )

    # Test query if provided
    if args.test_query:
        logger.info(f"\nTesting query: '{args.test_query}'")
        results = pipeline.query_collection(args.test_query)
        if results and "documents" in results:
            result_docs = results["documents"][0]
            logger.info("Found %s results:", len(result_docs))
            for i, doc in enumerate(result_docs[:3]):  # Show top 3
                preview = doc[:200]
                logger.info("Result %s: %s...", i + 1, preview)

    logger.info("Pipeline completed successfully!")


if __name__ == "__main__":
    main()

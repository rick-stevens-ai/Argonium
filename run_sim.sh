python similarity_analyzer.py chunks/modernbert_chunks --model scout --sample 4000 --embedding-model sentence-transformers:all-MiniLM-L6-v2 --multi-cluster 20 --generate-tsne --spatial-clustering --similarity-count 20 --output-pdf modernbert_chunks_4000_20_20.pdf

python similarity_analyzer.py chunks/pubmedbert_chunks --model scout --sample 4000 --embedding-model sentence-transformers:all-MiniLM-L6-v2 --multi-cluster 20 --generate-tsne --spatial-clustering --similarity-count 20 --output-pdf pubmedbert_chunks_4000_20_20.pdf

python similarity_analyzer.py chunks/sfr_chunks --model scout --sample 4000 --embedding-model sentence-transformers:all-MiniLM-L6-v2 --multi-cluster 20 --generate-tsne --spatial-clustering --similarity-count 20 --output-pdf sfr_chunks_4000_20_20.pdf

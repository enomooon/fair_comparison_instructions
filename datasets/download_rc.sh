# download datasets and pre-process it for the RC task.

# MARC is published on huggingface. So, you don't need download it here.

# NSMC for Korean
git clone https://github.com/e9t/nsmc.git
python sample_nsmc.py

# Prdect-ID for Indonesian
mkdir Prdect-ID
cd Prdect-ID
wget -O dataset.csv https://data.mendeley.com/public-files/datasets/574v66hf2v/files/f258d159-c678-42f1-9634-edf091a0b1f3/file_downloaded
cd ../
python sample_prdect-id.py

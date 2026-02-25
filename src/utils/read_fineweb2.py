from datatrove.pipeline.readers import ParquetReader

# limit determines how many documents will be streamed (remove for all)
# this will fetch the Italian filtered data
data_reader = ParquetReader("hf://datasets/HuggingFaceFW/fineweb-2/data/ita_Latn/train", limit=3) 
for document in data_reader():
    # do something with document
    print(document)
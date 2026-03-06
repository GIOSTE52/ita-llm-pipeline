from datatrove.pipeline.base import PipelineStep
from datatrove.data import DocumentsPipeline

class ItalianFeatureExtractor(PipelineStep):
    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        for doc in data:
            text = doc.text
            
            length = len(text)
            n_words = len(text.split())
            digit_ratio = sum(c.isdigit() for c in text) / max(length, 1)
            uppercase_ratio = sum(c.isupper() for c in text) / max(length, 1)
            punctuation_ratio = sum(c in ".,;:!?()" for c in text) / max(length, 1)
            whitespace_ratio = sum(c.isspace() for c in text) / max(length, 1)
            
            # naive type-token ratio
            tokens = text.split()
            unique_tokens = set(tokens)
            type_token_ratio = len(unique_tokens) / max(len(tokens), 1)

            doc.metadata.update({
                "length": length,
                "n_words": n_words,
                "digit_ratio": digit_ratio,
                "uppercase_ratio": uppercase_ratio,
                "punctuation_ratio": punctuation_ratio,
                "whitespace_ratio": whitespace_ratio,
                "type_token_ratio": type_token_ratio,
            })
            
            yield doc
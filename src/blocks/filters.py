from typing import List, Callable, Tuple
from datatrove.data import Document, DocumentsPipeline
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter

class ItalianFilter(BaseFilter):

    def __init__(self, filter_function: Callable[[Document], bool], exclusion_writer: DiskWriter = None, batch_size: int = 1):
        super().__init__(exclusion_writer, batch_size)
        self.filter_function = filter_function

    def filter(self, doc: Document) -> bool | Tuple[bool, str]:
        return self.filter_function(doc)

    def filter_batch(self, batch: List[Document]) -> List[bool | Tuple[bool, str]]:
        return super().filter_batch(batch)
        
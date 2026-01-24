import os
import argparse
from datatrove.data import Document
from datatrove.pipeline.filters import SamplerFilter
from datatrove.pipeline.writers import JsonlWriter
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import JsonlReader

#Scrivo un piccolo script di esempio per provare la pipeline con dei Document scitti manualmente

parser = argparse.ArgumentParser(description="ITA LLM Pipeline")
parser.add_argument(
    "--root-dir",
    type=str,
    default=os.path.expandvars("$HOME/ita-llm-pipeline"),
    help="Insert path of the project root directory"
)
parser.add_argument(
    "--output-dir",
    type=str,
    default=os.path.expandvars("$HOME/data"),
    help="Insert path for the output data"
)

args = parser.parse_args()

ROOT_DIR = args.root_dir
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUTPUT_DIR = args.output_dir

# Crea la cartella output se non esiste
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Scrivo un esempio di esecuzione con il SampleFilter(randomly keep 'rate'*100 percent of sample)
pipeline = [
    [
        Document(text="Qualsiasi cosa in italiano", id="0"),
        Document(text="Qualsiasi altra cosa in italiano", id="1"),
        Document(text="Ancora altre cose in lingua italiana, me so rotto 'e scatole de guardà li schermi", id="2"),
    ],
    SamplerFilter(rate=0.5),
    JsonlWriter(
        output_folder = "/home/stefano/test_data"
    )
]

#Eseguo la pipeline sopra definita
executor = LocalPipelineExecutor(
    pipeline = pipeline,
    tasks = 1,
    workers = 1
)
executor.run()

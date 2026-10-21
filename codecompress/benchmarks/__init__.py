"""Benchmark registry.

Each benchmark defines: data loading, context splitting, prompt construction,
generation config, and scoring metrics. See base.py for the interface.
"""

from codecompress.benchmarks.lca_summarization import LCASummarizationBenchmark
from codecompress.benchmarks.lcc import LCCBenchmark
from codecompress.benchmarks.longcodeqa import LongCodeQABenchmark
from codecompress.benchmarks.repobench import RepoBenchBenchmark

BENCHMARKS = {
    "lca_summarization": LCASummarizationBenchmark,
    "lcc": LCCBenchmark,
    "longcodeqa": LongCodeQABenchmark,
    "repobench": RepoBenchBenchmark,
}

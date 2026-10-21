"""Pipeline registry.

Each pipeline defines how to compress/render context and how to generate
completions. See text.py and image.py for implementations.
"""

from codecompress.pipelines.image import ImagePipeline
from codecompress.pipelines.text import TextPipeline

PIPELINES = {
    "text": TextPipeline,
    "image": ImagePipeline,
}

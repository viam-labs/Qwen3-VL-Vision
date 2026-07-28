"""
This file registers the model with the Python SDK.
"""

from viam.services.vision import Vision
from viam.resource.registry import Registry, ResourceCreatorRegistration

from .qwen3_vl import qwen3_vl


Registry.register_resource_creator(
    Vision.API,
    qwen3_vl.MODEL,
    ResourceCreatorRegistration(qwen3_vl.new, qwen3_vl.validate),
)

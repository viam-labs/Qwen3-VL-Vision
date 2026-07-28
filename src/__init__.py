"""
This file registers the model with the Python SDK.
"""

from viam.services.vision import Vision
from viam.resource.registry import Registry, ResourceCreatorRegistration

from .qwen import qwen


Registry.register_resource_creator(
    Vision.API,
    qwen.MODEL,
    ResourceCreatorRegistration(qwen.new, qwen.validate),
)

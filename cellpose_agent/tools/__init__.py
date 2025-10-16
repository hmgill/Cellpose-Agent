from .segmentation import (
    get_segmentation_parameters,
    run_cellpose_sam,
    refine_cellpose_sam_segmentation
)
from .search import (
    list_all_collections,
    search_documentation_vector,
    search_knowledge_graph,
    hybrid_search,
    get_parameter_relationships,
)
from .safety import (
    check_text_safety,
    check_image_safety,
    check_combined_safety,
    perform_visual_inspection
)

# Tools available to the safety agent
safety_tools = [
    check_text_safety,
    check_image_safety,
    check_combined_safety,
    perform_visual_inspection
]

# Tools available to the main cellpose agent
cellpose_tools = [
    get_segmentation_parameters,
    run_cellpose_sam,
    refine_cellpose_sam_segmentation,
    list_all_collections,
    search_documentation_vector,
    search_knowledge_graph,
    hybrid_search,
    get_parameter_relationships,
]

# All tools (for reference/testing)
all_tools = safety_tools + cellpose_tools

__all__ = [
    # tool sets
    "all_tools",
    "safety_tools",
    "cellpose_tools",
    # Segmentation tools
    "get_segmentation_parameters",
    "run_cellpose_sam",
    "refine_cellpose_sam_segmentation",
    # Search tools
    "list_all_collections",
    "search_documentation_vector",
    "search_knowledge_graph",
    "hybrid_search",
    "get_parameter_relationships",
    # Safety tools
    "check_text_safety",
    "check_image_safety",
    "check_combined_safety",
    "perform_visual_inspection",
]

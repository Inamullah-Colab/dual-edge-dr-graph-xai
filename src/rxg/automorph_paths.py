from __future__ import annotations

# AutoMorph M2 skeleton folders used for X1. These are deliberately skeleton-only.
M2_MAP_FOLDERS: dict[str, tuple[str, str]] = {
    "artery_global_skeleton": ("M2/artery_vein/artery_binary_skeleton", "x1_artery_global"),
    "vein_global_skeleton": ("M2/artery_vein/vein_binary_skeleton", "x1_vein_global"),
    "binary_global_skeleton": ("M2/binary_vessel/binary_skeleton", "x1_binary_global"),
    "artery_zone_b_skeleton": ("M2/artery_vein/macular_Zone_B_centred_artery_skeleton", "x1_artery_zone_b"),
    "vein_zone_b_skeleton": ("M2/artery_vein/macular_Zone_B_centred_vein_skeleton", "x1_vein_zone_b"),
    "binary_zone_b_skeleton": ("M2/binary_vessel/macular_Zone_B_centred_binary_skeleton", "x1_binary_zone_b"),
    "artery_zone_c_skeleton": ("M2/artery_vein/macular_Zone_C_centred_artery_skeleton", "x1_artery_zone_c"),
    "vein_zone_c_skeleton": ("M2/artery_vein/macular_Zone_C_centred_vein_skeleton", "x1_vein_zone_c"),
    "binary_zone_c_skeleton": ("M2/binary_vessel/macular_Zone_C_centred_binary_skeleton", "x1_binary_zone_c"),
}

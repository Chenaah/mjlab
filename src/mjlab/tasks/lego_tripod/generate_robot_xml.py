#!/usr/bin/env python3
"""Generate the clean robot XML for the mjlab LegoTripod task.

Uses LegoRobotBuilder (create_tripod) with restructure=True to produce a proper
nested body hierarchy (no weld constraints, single freejoint).  The resulting XML
is then stripped of floor / scene elements so mjlab can load it as a stand-alone
entity.

Output:
    mjlab/src/mjlab/tasks/lego_tripod/assets/lego_tripod.xml

Usage (run from metamachine_open/):
    conda activate mm
    python mjlab/src/mjlab/tasks/lego_tripod/generate_robot_xml.py
"""

import sys
import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent.parent
print(f"[generate] project root: {PROJECT_ROOT}")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PRIVATE_PLUGINS_DIR = str(PROJECT_ROOT / "metamachine_plugins" / "private_plugins")
OUTPUT_PATH = Path(__file__).parent / "assets" / "lego_tripod.xml"

# Robot parameters from lego_tripod_turn.yaml
_THIGH_LENGTH = 0.24
_CALF_LENGTH  = 0.24
_BELLY_MASS   = 0.256
_STICK_MASS   = 0.075


def generate():
    print("[generate] Loading lego_legs plugin...")
    from metamachine.robot_factory import load_plugins_from
    load_plugins_from(PRIVATE_PLUGINS_DIR)

    from metamachine_plugins.private_plugins.lego_legs.lego_robot_builder import LegoRobotBuilder

    print("[generate] Building tripod robot...")
    builder = LegoRobotBuilder()

    # Ball modules (hip joints)
    for i in range(3):
        builder.add_ball(f"m{i}")

    # Central body
    builder.add_adaptor("belly", mass=_BELLY_MASS)

    # Thigh and calf sticks
    for i in range(3):
        builder.add_stick(f"thigh{i}", length=_THIGH_LENGTH, mass=_STICK_MASS)
    for i in range(3):
        builder.add_stick(f"calf{i}", length=_CALF_LENGTH, mass=_STICK_MASS)

    # Weld connections (same as in lego_tripod_turn.yaml)
    weld_pairs = []
    for i in range(3):
        weld_pairs.append((f"dock-belly-{i}f",   f"dock-thigh{i}-1m"))
        weld_pairs.append((f"dock-m{i}-0f",       f"dock-thigh{i}-0m"))
        weld_pairs.append((f"dock-calf{i}-0m",    f"dock-m{i}-1f"))
    builder.weld(weld_pairs)

    # Save to a temp file first, then restructure into a nested hierarchy
    with tempfile.TemporaryDirectory() as tmp_dir:
        flat_path = Path(tmp_dir) / "lego_tripod_flat.xml"
        print(f"[generate] Saving flat XML to temp dir...")
        # save() with restructure=True calls restructure_modular_robot internally
        restructured_path = builder.save(str(flat_path), restructure=True)
        print(f"[generate] Restructured XML at: {restructured_path}")

        print("[generate] Stripping scene elements...")
        _strip_robot_xml(Path(restructured_path), OUTPUT_PATH)

    print(f"[generate] ✓ Clean restructured robot XML written to:\n  {OUTPUT_PATH}")


def _strip_robot_xml(src: Path, dst: Path) -> None:
    """Parse the restructured MetaMachine XML and write a robot-only version.

    Removes:
    - floor geom / lights / scene cameras in worldbody
    - scene-only textures and materials (matplane, hfield, boundary)
    - option / size / visual blocks (mjlab scene handles those)
    - actuators (IdealPdActuatorCfg adds its own; keeping XML motors would
      create duplicates)

    Keeps:
    - compiler / default (angle / inertia settings)
    - asset: mesh + robot materials
    - worldbody robot body tree (the restructured hierarchy has a single root body)
    - contact exclusions
    - sensor (IMU quaternion + gyro per module)
    """
    tree = ET.parse(str(src))
    root = tree.getroot()

    new_root = ET.Element("mujoco")
    new_root.set("model", "lego_tripod")

    # compiler
    compiler = root.find("compiler")
    if compiler is not None:
        new_root.append(compiler)

    # default — fix conaffinity so robot geoms can collide with terrain
    # The MetaMachine default sets conaffinity="0" which prevents floor contact
    # when mjlab adds terrain separately.  We change it to "1" here.
    default = root.find("default")
    if default is not None:
        for geom_default in default.iter("geom"):
            if geom_default.get("conaffinity") == "0":
                geom_default.set("conaffinity", "1")
        new_root.append(default)

    # asset — keep meshes and robot materials, drop scene-only items
    _SKIP_MATERIALS = {"matplane", "hfield", "boundary"}
    _SKIP_TEXTURES  = {"texplane", "boundary"}
    asset_src = root.find("asset")
    if asset_src is not None:
        asset_dst = ET.SubElement(new_root, "asset")
        for child in asset_src:
            if child.tag == "mesh":
                asset_dst.append(child)
            elif child.tag == "material":
                if child.get("name") not in _SKIP_MATERIALS:
                    asset_dst.append(child)
            elif child.tag == "texture":
                name = child.get("name", "")
                typ  = child.get("type", "")
                if name not in _SKIP_TEXTURES and typ != "skybox":
                    asset_dst.append(child)
        if len(asset_dst) == 0:
            new_root.remove(asset_dst)

    # worldbody — drop floor geom and scene lights, keep robot bodies
    worldbody_src = root.find("worldbody")
    if worldbody_src is not None:
        worldbody_dst = ET.SubElement(new_root, "worldbody")
        for child in worldbody_src:
            if child.tag == "geom" and child.get("name") == "floor":
                continue
            if child.tag == "light":
                continue
            worldbody_dst.append(child)

    # actuator — intentionally omitted; IdealPdActuatorCfg will add its own
    # motors via edit_spec().  Including them here would produce duplicates.

    # contact exclusions
    contact = root.find("contact")
    if contact is not None:
        new_root.append(contact)

    # sensors (imu_quat*, imu_gyro* per module)
    sensor = root.find("sensor")
    if sensor is not None:
        new_root.append(sensor)

    dst.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(new_root, space="  ")
    ET.ElementTree(new_root).write(str(dst), encoding="unicode", xml_declaration=False)


if __name__ == "__main__":
    generate()

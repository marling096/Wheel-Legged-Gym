# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Parse wl.urdf for lumped masses / inertia used by simplified controllers."""

from __future__ import annotations

import xml.etree.ElementTree as ET


def parse_wheellegged_urdf_inertial(urdf_path: str) -> dict:
    """Extract totals from ``resources/robots/wl/urdf/wl.urdf`` style URDF."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    total_mass = 0.0
    base_mass = None
    base_inertia = {}
    wheel_mass_sum = 0.0
    wheel_radius = None

    for link in root.findall("link"):
        name = link.get("name") or ""
        inertial = link.find("inertial")
        if inertial is None:
            continue
        mass_el = inertial.find("mass")
        if mass_el is None:
            continue
        m = float(mass_el.attrib["value"])
        total_mass += m

        inertia_el = inertial.find("inertia")
        inertia = {}
        if inertia_el is not None:
            for key in ("ixx", "iyy", "izz"):
                if key in inertia_el.attrib:
                    inertia[key] = float(inertia_el.attrib[key])

        if name == "base_link":
            base_mass = m
            base_inertia = inertia
        if "wheel" in name.lower():
            wheel_mass_sum += m

        cyl = link.find("./collision/geometry/cylinder")
        if cyl is not None and "wheel" in name.lower() and "radius" in cyl.attrib:
            wheel_radius = float(cyl.attrib["radius"])

    return {
        "total_mass": total_mass,
        "base_mass": base_mass or 0.0,
        "base_inertia_ixx": base_inertia.get("ixx"),
        "base_inertia_iyy": base_inertia.get("iyy"),
        "base_inertia_izz": base_inertia.get("izz"),
        "wheel_mass_sum": wheel_mass_sum,
        "wheel_radius": wheel_radius,
    }

"""ORCS's privileged SONIC adapters on a G1: dodging thrown balls, and climbing a box and
walking over curbs it reads with a height scan. See ``README.md``."""

from __future__ import annotations

from pathlib import Path

import mjswan
import onnx

from . import upstream

#: (policy, task), in the order upstream registers them.
SCENES = (
    ("Dodge", upstream.DODGE),
    ("Climb", upstream.PERLOCO_OMRE),
    ("Curb", upstream.PERLOCO_GRAIL),
)
#: The attribution the checkpoints' license requires beside its text.
CHECKPOINT_NOTICE = (
    "Licensed by NVIDIA Corporation under the NVIDIA Open Model License."
)
#: The clip's dataset card declares MIT and names no copyright holder.
OMNI_NOTICE = """\
OmniRetarget Dataset: https://huggingface.co/datasets/omniretarget/OmniRetarget_Dataset at
1352591, the robot-terrain clip climb_16_z_scale_1.0 and its boxes, MIT as the dataset
card declares. Lujie Yang, Xiaoyu Huang, Zhen Wu, Angjoo Kanazawa, Pieter Abbeel, Carmelo
Sferrazza, C. Karen Liu, Rocky Duan, Guanya Shi. OmniRetarget: Interaction-Preserving Data
Generation for Humanoid Whole-Body Loco-Manipulation and Scene Interaction. 2025.
"""


def setup_builder() -> mjswan.Builder:
    roots = upstream.resolve_roots()
    upstream.register_tasks(roots)
    from . import terms

    builder = mjswan.Builder()
    project = builder.add_project(name="ORCS", license=roots["orcs"] / "LICENSE")
    project.set_notice(roots["orcs"] / "NOTICE")
    for name, task_id in SCENES:
        env_cfg = terms.browser_env_cfg(upstream.play_env_cfg(roots, task_id))
        policy, contract = upstream.ensure_policy(roots, task_id, env_cfg)
        scene = project.add_scene_mjlab(task_id, env_cfg=env_cfg)
        # mjswan matches the G1 by name, and these task ids do not name it.
        scene.add_attribution(
            "unitree_g1",
            license="BSD-3-Clause",
            copyright='HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics")',
        )
        scene.add_attribution(
            "orcs-checkpoints",
            license=upstream.checkpoint_license(),
            notice=CHECKPOINT_NOTICE,
        )
        if task_id == upstream.PERLOCO_OMRE:
            scene.add_attribution("omniretarget", notice=OMNI_NOTICE)
        if task_id == upstream.PERLOCO_GRAIL:
            scene.add_attribution(
                "grail", license=Path(__file__).with_name("LICENSE.grail")
            )
        scene.add_policy(
            name=name,
            policy=onnx.load(str(policy)),
            # The released actor's own input order.
            in_keys=["tokenizer", "actor", "augmentation"],
            **contract,
        )
    return builder

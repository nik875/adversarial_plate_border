# SPAR: Street-Legal Physical Adversarial Attacks on Automatic License Plate Recognition Systems

Official repository for the paper **"SPAR: Street-Legal Physical Adversarial Attacks on Automatic License Plate Recognition Systems"**, accepted at the SiMLA 2026 Workshop.

Paper Link: https://arxiv.org/abs/2604.02457v1

## Overview

Automatic License Plate Recognition (ALPR) systems are widely deployed in law enforcement, tolling, and traffic monitoring. While previous research has demonstrated vulnerabilities to adversarial examples, most attacks assume unrealistic threat models, require direct modification of license plate characters, or fail to transfer reliably to the physical world.

We introduce **SPAR (Street-Legal Physical Adversarial Rim)**, a physically realizable adversarial attack that modifies only the border region surrounding a license plate while leaving the plate characters unchanged. SPAR is designed under realistic attacker constraints, including limited resources and legal restrictions on license plate modifications.

## Key Contributions

* Introduce **SPAR**, a border-based adversarial attack that preserves license plate legibility.
* Demonstrate effective attacks against ALPR systems under a realistic white-box threat model.
* Incorporate homography-based transformations during optimization to improve robustness to viewpoint and distance changes.
* Show that Total Variation (TV) regularization produces structured perturbations that transfer more effectively to physical environments.
* Evaluate attacks in both digital and real-world settings.
* Investigate the role of large language models (LLMs) in assisting attack design and iteration.

## Results

SPAR successfully transfers from digital optimization to physical deployment and significantly degrades ALPR performance while maintaining a visually unobtrusive appearance.

Key findings include:

* Significant reduction in ALPR recognition accuracy.
* Successful targeted impersonation attacks in both digital and physical evaluations.
* Improved robustness through structured, low-frequency perturbations.
* Evidence that LLM assistance lowers the barrier to developing effective physical attacks.

## Disclaimer

This work is intended solely for academic research, security evaluation, and the development of more robust machine learning systems. The techniques described should only be used in authorized and ethical research settings.


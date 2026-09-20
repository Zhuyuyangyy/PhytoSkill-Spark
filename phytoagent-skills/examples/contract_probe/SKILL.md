---
name: contract-probe
description: Validate the Phyto Skill SDK contract using an explicitly synthetic fixture. Use only for SDK integration checks.
---

Pass a nonempty `message` to exercise input validation, execution and output
validation. The output echoes the message and declares `synthetic_fixture`.
This is an SDK example; it does not analyze images, plants or environmental data.

Only `fixture` mode is supported. See [schema.json](schema.json) for the input
and output contract. The Python entrypoint is [skill.py](skill.py).

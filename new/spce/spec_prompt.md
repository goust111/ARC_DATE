You are tasked with extracting a detailed **input/output specification** for ARC-AGI-2 task `{task_id}`. This spec will be used to:
1. Guide a separate LLM agent to write a `generate_input()` function
2. Allow a **human** to quickly review and correct the spec before generation

## Your inputs

1. **Solver code**: Read `{solver_path}`
2. **Eval I/O data**: Read `{eval_path}` — contains training and test examples (input/output pairs).
3. **Task annotation**:
```json
{annotation_json}
```

## Your task

Produce a comprehensive **input/output spec** in YAML format that captures ALL structural properties and statistical patterns a generator must replicate. This spec should be detailed enough that:
- A generator author can implement `generate_input()` purely from this spec
- A reviewer can verify generated data quality by checking against this spec

## Requirements

1. **Read the solver code** carefully to understand what structures it expects and handles (including edge cases, skip/continue branches).
2. **Systematically analyze ALL eval examples** (both train and test) using Python scripts. Don't just read one example — analyze all of them and look for patterns, variations, and edge cases. For each example, extract concrete measurements.
3. **Analyze BOTH inputs and outputs**: compare what's in the input vs what's in the output. What gets added, removed, or changed?
4. **For every property that varies between examples**, analyze its distribution and express the result as a **range** in the spec — don't list raw observed values or reference specific examples (e.g. "train_0") in the final YAML. The spec is not for describing existing eval I/O data, but for guiding the generation of **new** data of the task.
5. **Analyze object-level properties**: for each sprite/object in each example, compute structural properties (e.g. shape, size, connectivity, symmetry, color constraints — whatever is relevant to the task).

### Mandatory eval data analysis (do NOT skip)

Before writing the spec, you MUST use Python scripts to compare input and output grids of every eval example, extracting concrete measurements for all observable structural properties of objects and their relationships. **The solver defines what transformations are valid, but eval data often enforces stricter constraints that the solver merely tolerates** (e.g. the solver handles partial overlap, but eval data always shows full-edge contact). Document these data distribution constraints in the spec — they are critical for generating realistic inputs.

After drafting the spec, verify: "Could a generator satisfying all spec constraints still produce inputs visibly different from eval data?" If yes, the spec is incomplete.

## Output format requirements

The YAML should have two sections:

1. **`spec:`** — the task specification. Use terse natural-language summaries with embedded semantics (e.g. `count: "2–4 matching + 0–1 distractors"`) rather than raw observed value lists. Include ASCII diagrams for key structures where helpful. Try to align objects/structures with tags in task annotation (e.g. src, dst, path, trajectory) whenever possible. Target ≤30 lines.
2. **`generation_strategy:`** — describe how to construct valid inputs. Prefer an "output-first, then invert" approach: build the full output first (typically the more constrained structure), then derive the input (the easier thing) from it by undoing the solver's transformation described in `main()`. This ensures tight constraints are satisfied by construction rather than by random trial, which often has near-zero acceptance rate. Examples of output-first strategy by task type:
   - **move_to_match**: build the assembled output (dst with src attached/plugged/filled/merged to it), then detach/unplug/split and scatter the src to get the input.
   - **move_along_path**: build the complete trajectory in the output, then place the src and any bouncing/turning structures, and remove the trajectory to get the input.

   **Critical**: every conditional behavior in `transformation:` (e.g. "if flip then swap colors", "if X then Y") MUST appear as an explicit numbered step in `generation_strategy:`, not just described in the spec section. The generation_strategy is the authoritative checklist that the generator agent will implement step-by-step — anything not listed as a step will likely be skipped. Similarly, any uniqueness/disambiguation constraint required by the solver (e.g. "only one color can match the cavity size") must be an explicit step or constraint in generation_strategy, not left implicit.

Aim for **≤40 lines** total. The spec should be concise enough for a human to review at a glance.

**YAML style**: Do not quote string values unless YAML syntax requires it. Write `grid: 10-22 rows x 10-22 cols` not `grid: "10-22 rows x 10-22 cols"`. Use `|` for multi-line blocks. Use nested YAML keys to organize properties by component (e.g. `sprite:`, `frame:`, `ray:` as top-level keys under `spec:`, with sub-keys for each property). Do NOT write `spec:` or `generation_strategy:` as a single `|` text block.

**Distinguishing required vs observed constraints**: If a property is NOT required by the solver but is a consistent pattern in the eval data (e.g. all sprites happen to be symmetric), mark it with `[observed]` in the spec. Example: `symmetry: LR and/or UD [observed]`. This tells the generator author it's a nice-to-have, not a hard requirement. Conversely, if the solver hard-codes or asserts a value, it is required — do NOT mark it `[observed]`.

**Constraints should be quantifiable and verifiable**: prefer `spanning >=75% of rows and columns` over `meanders through most of the grid`. The generator's quality audit will programmatically check each constraint — vague descriptions can't be checked.

IMPORTANT: Use the Python interpreter at `python` for ALL analysis. Write scripts to systematically extract properties from eval data. Be thorough — check properties you might not initially think are important (symmetry, regularity patterns, color relationships, etc.)

Return ONLY the final YAML spec.

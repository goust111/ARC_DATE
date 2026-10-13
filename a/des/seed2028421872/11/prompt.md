你是一个设计 ARC-AGI（抽象与推理语料库，Abstraction and Reasoning Corpus）谜题的专家。你的任务是对下方提供的谜题逻辑与格式进行深入分析。通过参考示例谜题的推理与概念，你需要运用发散性思维进行创新扩展。请发挥创造力，构思全新且巧妙的推理谜题。
下面给出 20 个已有 ARC 题目的描述作为参考。每个描述通常包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="train" puzzle_id="668eec9a" lines="29-47">
<description>
Summarize the foreground line objects as ordered horizontal color stripes.

Input structure:
- A dominant background fills a rectangular grid.
- Each non-background color forms one contiguous one-cell-thick line segment.
- The line segments have distinct topmost rows, and there are at most five of
  them.

Rule:
- Sort the foreground colors by the topmost row occupied by their line
  segment.
- Create a 5x3 output filled with the background color.
- Place one solid 1x3 row for each foreground color, preserving that
  top-to-bottom order and padding any unused rows above with background.

Key insight: The exact line length and slope do not matter; only the
vertical order of the colored line segments is encoded in the output stack.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="train" puzzle_id="689c358e" lines="72-88">
<description>
Mark each arrow-like sprite's pointed frame endpoint with its color and the opposite endpoint black.

Input structure:
- The grid has a one-pixel solid rectangular frame around a dominant interior background.
- Inside the frame are monochrome six-pixel sprites shaped as asymmetric T arrows.
- Each sprite has a length-4 bar and a length-3 crossbar one cell closer to the pointed end.

Rule:
- For each sprite, follow its bar axis from the shorter end to the frame.
- Paint that frame endpoint with the sprite color.
- Paint the opposite endpoint on the same row or column black.
- Leave the interior background, sprites, and all other frame pixels unchanged.

Key insight:
- The direction is encoded by the crossbar's off-center position: the nearer end is the pointing end.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="train" puzzle_id="7d7772cc" lines="70-92">
<description>
Move each source pixel toward the separator when it matches its aligned reference pixel, otherwise to the far edge.

Input structure:
- A single full separator line is part of a rectangular frame; the frame's
  extra cells mark the reference side.
- On the reference side, one row or column of colored pixels is aligned
  with one row or column of source pixels on the opposite side.
- Each side has its own dominant background color.

Rule:
- Compare each source pixel with the reference pixel on the same row or column.
- If their colors match, place the source pixel in the cell adjacent to the
  separator on the source side.
- If their colors differ, place the source pixel on the far outer edge along
  the same row or column.
- Erase the original source pixels to the source-side background; keep the
  frame and reference pixels unchanged.

Key insight:
The extra frame cells identify which side is the reference; equality with the
aligned reference pixel decides whether each source pixel moves inward or outward.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="train" puzzle_id="ecb67b6d" lines="138-155">
<description>
Recolor the grey diagonal-chain regions in an orange field to teal.

Input structure
The input contains only grey pixels and orange pixels.  Grey pixels form
several 8-connected regions embedded in the orange field.

Rule
Decompose each grey 8-connected region into 4-connected lobes.  Recolor the
whole region teal only when it contains a straight diagonal run of at least
three pixels and those lobes meet only by diagonal corner contacts in a
non-branching diagonal chain or cycle; a single 4-connected region is
recolored only when it is a square diagonal staircase.

Key insight
The teal regions are not selected by size or position; they are the grey
regions whose internal connectivity depends on diagonal corner joins, while
same-edge cracks, rectangles, bars, and branching/noisy regions remain grey.
</description>
</reference_puzzle>

<reference_puzzle index="5" source="train" puzzle_id="52df9849" lines="49-67">
<description>
Complete each partially hidden simple foreground shape by recoloring the occluding pixels in its footprint.

Input structure
- The grid has a dominant background and several solid foreground shapes.
- Some shapes are partially hidden by other foreground colors, but their full
  footprint is still indicated by a foreground-filled bbox or by a foreground-
  filled 45-degree diagonal line.

Rule
- For each foreground color, complete its full rectangular bbox when that bbox
  contains no background and includes other colors.
- Otherwise, complete its 45-degree diagonal line when all visible pixels lie
  on that line and the hidden line cells are foreground, not background.
- Recolor only those hidden footprint pixels; leave all other pixels unchanged.

Key insight
- The occluded cells are exactly the non-background cells inside a simple
  shape's implied footprint, so completing shapes is a local recoloring problem.
</description>
</reference_puzzle>

<reference_puzzle index="6" source="train" puzzle_id="aaecdb9a" lines="10-26">
<description>
Convert scattered foreground objects into bottom-aligned count bars in a fixed five-color order.

Input structure
The grid has a dominant background color and foreground objects in the task
palette: grey, red, teal, maroon, and pink. Objects are monochromatic
8-connected components; some palette colors may be absent.

Rule
For each palette color in order grey, red, teal, maroon, pink, count its
8-connected objects. Create a background grid with five columns and height
equal to the largest count, then fill each column upward from the bottom
with its palette color for that count.

Key insight
The output does not preserve object shapes or positions; it is a histogram
of per-color 8-connected object counts, using the learned palette order.
</description>
</reference_puzzle>

<reference_puzzle index="7" source="train" puzzle_id="1be83260" lines="41-58">
<description>
Decode the colored slot map inside the marked motif and rebuild the motif lattice with those colors.

Input structure:
- A blank background surrounds a rectangular lattice of identical motifs.
- The motif rows and columns are separated by one background line.
- One motif has colored marker pixels in its internal odd-row/odd-column slots.

Rule:
- The motif size is 2R+1 by 2C+1, where R by C is the motif lattice size.
- Read the marked motif's odd internal slots as the color map for the R by C output motifs.
- Slots that still contain the repeated motif color map to that repeated motif color.
- Rebuild the compact lattice on a canvas filled with the repeated motif color,
  painting each motif mask with its decoded color.

Key insight:
- The holes and filled odd slots inside one motif are a miniature legend for recoloring the entire repeated motif lattice.
</description>
</reference_puzzle>

<reference_puzzle index="8" source="eval" puzzle_id="4c416de3" lines="99-118">
<description>
Replicate a pre-drawn corner pattern to every other marker-bearing corner.

Input structure:
- bg = most frequent color. Hollow 0-frames (possibly partial L-shapes); each
  frame-corner cell is either 0 or a painted (non-bg, non-0) color.
- >= 1 'source' corner: the pattern is already drawn (one color, both outward
  cells and an inward-diagonal trail).
- Other corners: a single colored pixel on the inward diagonal at the pattern's
  deepest (k,k) offset; the pixel's color is that corner's target color.

Rule:
- Extract the canonical (TL-oriented) pattern from any source.
- For each non-source corner whose marker cell is colored, paint the same
  pattern rotated to that corner's orientation, using the marker's color.

Key insight: the pattern is symmetric about the corner's inward-diagonal axis
(canonical dr == dc), so it reflects cleanly across all four corner orientations.
Painting uses each target's OWN marker color, not the source color.
</description>
</reference_puzzle>

<reference_puzzle index="9" source="train" puzzle_id="e84fef15" lines="39-59">
<description>
Reconstruct the repeated tile from its separated copies and mark the missing pixels in the damaged copy blue.

Input structure:
- The input is a lattice of equal-size blocks separated by one-cell solid rows
  and columns of the same separator color.
- All blocks are copies of one motif except exactly one block, where some
  motif pixels have been overwritten by the motif background color.
- Blue is absent from the input and is reserved as the output marker color.

Rule:
- Recover the motif by taking the unique majority color at each local block
  coordinate across all block copies.
- Find the single block that differs from this consensus motif.
- Return the consensus motif, recoloring the differing local coordinates blue.

Key insight:
- The repeated copies provide redundancy: the damaged positions are exactly
  the coordinates where the lone nonconforming copy has background instead of
  the consensus motif color.
</description>
</reference_puzzle>

<reference_puzzle index="10" source="eval" puzzle_id="8e5c0c38" lines="9-21">
<description>
Clean each color layer by enforcing vertical mirror symmetry.

Input structure:
- bg = background. Several non-bg colors, each forming a nearly-symmetric pattern about some vertical axis.
- Some pixels in each color are "noise" that breaks the symmetry.

Rule:
- For each non-bg color, find the vertical axis that best explains its pixel layout.
- Remove any pixel that lacks a same-color mirror counterpart across that axis.

Key insight: Each color has its own independent symmetry axis; the task is per-color denoising, not whole-grid symmetry.
</description>
</reference_puzzle>

<reference_puzzle index="11" source="eval" puzzle_id="a25697e4" lines="9-23">
<description>
Dock movable tools into border-anchored frames by filling cavities.

Input structure:
- bg = most common color.
- Border-touching components cluster into anchor frames with interior bg cavities.
- Non-border 2-color components are movable tools.

Rule:
- Match each tool to a frame whose cavity size equals one of its color-region counts.
- Rotate/flip the tool so one color fills the cavity exactly; the other stays outside.
- If the docking orientation uses a reflection, swap the tool's two colors.

Key insight: Cavity size determines tool–frame pairing; reflection triggers color swap.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="train" puzzle_id="9caf5b84" lines="28-45">
<description>
Recolor every pixel whose color is not one of the two most frequent input colors to orange.

Input structure:
- The grid contains at least three colors.
- Orange is absent from the input and is used only as the output replacement color.
- Exactly two colors occupy the top two frequency ranks; they may tie with each
  other, but no third color ties them at the cutoff.

Rule:
- Preserve every pixel whose color is one of the two most frequent input colors.
- Replace every other pixel with orange.

Key insight:
- The visible distinction is not the numeric identity of the colors but their
  population: the two majority colors remain, while all minority colors merge
  into the new orange color.
</description>
</reference_puzzle>

<reference_puzzle index="13" source="eval" puzzle_id="5dbc8537" lines="10-23">
<description>
Fill rectangular cavities in a template shape by placing source sprites into matching holes.

Input structure:
- bg = most common color. A dominant "base" color forms a large frame/template with rectangular gaps.
- Small colored sprites sit outside the template region.

Rule:
- The base color (largest connected component excluding bg) defines the template bbox.
- Each rectangular hole inside the template matches exactly one outside sprite by size.
- Place each sprite into its matching cavity; output is the filled template bbox.

Key insight: Sprite bounding boxes match cavity dimensions exactly — it's a jigsaw-fit puzzle where each piece has a unique slot.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="train" puzzle_id="9f41bd9c" lines="10-30">
<description>
Move the side-attached foreground object to the opposite wall, shear its lower rows inward, and paint the crossed floor cells.

Input structure:
- The top row is the upper background color and the bottom row is the
  floor color.
- A single non-background, non-floor object touches exactly one vertical
  grid edge and rests immediately above the first solid floor row.

Rule:
- Clear the source object to the upper background color.
- Repaint each object row on the opposite side: the first three relative
  rows keep the edge alignment, and each lower row shifts one more pixel
  inward from that side.
- On the first floor row, repaint the traversed side-to-object segment
  maroon, stopping just before the final outer bottom support pixel.

Key insight:
- The side of the source object gives the horizontal direction, while the
  row index inside the object gives the one-cell-per-row diagonal shear.
</description>
</reference_puzzle>

<reference_puzzle index="15" source="eval" puzzle_id="291dc1e1" lines="7-22">
<description>
Reorder and stack rectangles by proximity to two colored edge lines.

Input structure:
- bg = most common color.
- Two edge lines (h + v) meet at a corner marked by a unique color.
- "Blue" line is color 1 if present, else the longer line; other is "red."
- Solid rectangles in the interior.

Rule:
- Sort rectangles by (distance to blue edge, distance to red edge).
- If the blue edge is vertical, rotate each rectangle 90° CCW.
- Stack top-to-bottom, centered horizontally.

Key insight: Edge-line colors encode a coordinate system; proximity to each axis determines stacking order and rotation.
</description>
</reference_puzzle>

<reference_puzzle index="16" source="train" puzzle_id="37ce87bb" lines="33-47">
<description>
Compute a grey result bar by adding teal bar heights and subtracting red bar heights.

Input structure
- The grid has a dominant background and several bottom-aligned one-cell-wide vertical bars.
- Teal bars are positive operands; red bars are negative operands.
- The penultimate column is an empty result slot.

Rule
- Keep every input bar unchanged.
- Compute result height = total teal bar height - total red bar height.
- Paint that many bottom cells of the penultimate column grey.

Key insight
- The bars encode arithmetic by height, with color providing the sign.
</description>
</reference_puzzle>

<reference_puzzle index="17" source="eval" puzzle_id="aa4ec2a5" lines="52-66">
<description>
Add red outlines around objects, and mark enclosed holes inside hollow objects.

Input structure:
- A mostly uniform background contains one or more 4-connected foreground objects.
- Some objects may enclose 4-connected background holes inside their bbox.

Rule:
- For every foreground component, draw a one-cell 8-neighborhood outline in red (2).
- If a component has enclosed background holes, recolor the component to teal (8).
- Fill those enclosed holes with magenta (6) before drawing the outline.

Key insight: Holes are treated as part of the object for outlining, while only objects
that actually enclose holes get the teal/magenta interior recoloring.
</description>
</reference_puzzle>

<reference_puzzle index="18" source="eval" puzzle_id="7ed72f31" lines="8-22">
<description>
Mirror colored sprites using red (2) markers as reflection axes.

Input structure:
- bg = most common color. Red (2) = marker color.
- Colored sprites appear near red markers.

Rule:
- Red line within a sprite (vertical or horizontal) → mirror payload across that axis.
- Single red pixel in a mixed sprite → point-reflect payload around it.
- Isolated red pixel (alone, no payload) → point-reflect each adjacent 8-connected color component.
- Originals remain; mirrored pixels overlay.

Key insight: Red marker geometry (line vs single pixel) determines reflection type; isolated vs embedded red determines what gets reflected.
</description>
</reference_puzzle>

<reference_puzzle index="19" source="eval" puzzle_id="b0039139" lines="29-44">
<description>
Recolor a cropped template and repeat it according to a component-count stencil.

Input structure:
- Solid blue (1) separator rows or columns split the grid into four sections.
- Section 1 is a black-background shape template; section 2 is a component-count stencil.
- Sections 3 and 4 are solid color swatches for output foreground and background.

Rule:
- Crop section 1 to the bbox of its non-black pixels and recolor it with the two swatches.
- Count 4-connected non-black components in section 2.
- Repeat the recolored crop that many times, inserting one background-colored row/column between copies.

Key insight: The second section encodes only the repeat count; the output orientation
is inherited from whether the input separators are columns or rows.
</description>
</reference_puzzle>

<reference_puzzle index="20" source="eval" puzzle_id="97d7923e" lines="81-94">
<description>
Recolor vertical bars by legend-driven rank selection.

Input structure:
- bg = background. Top-left: monochrome blocks touching row 0; each has color K and height k.
- Main grid: vertical bars capped top and bottom by a single color, with solid-color interiors.

Rule:
- For each legend entry (K, k), find vertical bars capped by K.
- Rank bars by interior length (longest = rank 1).
- Recolor the rank-k bar's interior to K.

Key insight: Legend height is a rank index into length-sorted bars, not a literal size.
</description>
</reference_puzzle>

你的任务：
第一步：分析上述参考题中涉及的基础操作，思考这些操作之间有哪些非显而易见的串联方式。重点思考：
 - 一个操作的输出如何以意想不到的形式成为另一个操作的输入条件。
 - 信息可以通过哪些不直觉的方式编码在网格中。
 - 寻找已有操作之间新的因果关系和信息传递路径。

第二步：基于上述思考，开启头脑风暴，运用发散性思维，用中文创造至少5个全新的 ARC 谜题描述，若有较多合适的创新，也可以生成更多，不设上限。严格使用下面的 XML风格格式输出：

<new_puzzle>
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。
</new_puzzle>

注意事项：
1. 任务不应依赖大量先验知识或复杂计算，而应依赖对模式、结构与空间关系的直觉推理。
2. 若涉及计算，只允许使用简单的计数、排序与数量比较；不允许使用复杂的数学公式或运算。
3. 问题必须能从少量示例中推断出来，但不代表题目一定简单。设计时需从人类解题者的视觉角度思考：规则涉及的所有关键要素（对象、关系、属性）必须在网格上肉眼可见，不应依赖人眼无法直接观察到的隐含信息或需要推算的隐藏属性。规则可以很难发现，难到反复对比所有样例仍需要深度思考才能归纳出来，但一旦发现后必须能用来解释所有样例且产出唯一确定的输出。
4. 生成的问题描述旨在促进后续用于创建题目网格的代码生成。因此，对输入/输出网格的描述以及规则总结必须极其详细；语言风格应易于理解；对于特定对象（如箭头），必须在描述中明确其绘制方式。
5. 描述谜题时避免使用 0 或 1 这类颜色编号；应使用颜色名称，例如 “red”。可用颜色如下：Black、Blue、Red、Green、Yellow、Grey、Magenta、Orange、Azure、Maroon。
6. 五个新题的核心推理机制应明显不同——即从观察样例到归纳规则的认知路径不同，而不仅仅是场景、对象或颜色不同。
7.  题目难度应来自机制的隐蔽性而非步骤数。关键信息难以准确定位，需要反复对比尝试不同的角度才能发现真正的规则。


生成后对生成的题执行以下四项审查（审查内容不需要保存到文件）：
1. 原创性
  新题不可与上述20道参考题相同或者是类似，不可只是替换了颜色、方向、对象类型、维度等表层要素。判断标准：新题的核心推理路径（解题者从样例中归纳规则的认知过程）应与任何一道参考题不同。
2. 确定性
  规则完全确定，同一输入只能产生唯一输出，每一步操作都有明确定义。
3. 可实现性
  所有必要细节都已给出——对象如何生成、如何识别、如何变换、边界情况如何处理，读完描述后可以直接写代码。
4. ARC 合规性
  网格尺寸合理（≤30×30），使用颜色不超过 10 种，规则复杂度适中，人类能从 3-5个样例中发现规律。
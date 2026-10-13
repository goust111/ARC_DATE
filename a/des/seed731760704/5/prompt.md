你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="eval" puzzle_id="dd6b8c4b" lines="9-22">
<description>
Move the closest reachable maroon pixels into the 3×3 block around the unique red center.

Input structure:
- bg = orange floor, magenta walls, scattered maroon (9) pixels.
- A unique red (2) pixel marks the center of a 3×3 destination block.

Rule:
- Maroon pixels connected to the destination through floor/maroon cells (not blocked by walls) are candidates.
- Rank candidates by distance to the destination; the closest ones fill the block in row-major order.
- Moved maroons are erased (replaced with floor color).

Key insight: Wall barriers restrict connectivity — only maroon pixels reachable via flood-fill through the floor qualify, so proximity alone isn't enough.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="eval" puzzle_id="9385bd28" lines="71-87">
<description>
Fill rectangles between diagonal marker pairs using a color legend.

Input structure:
- bg grid with a vertical legend at the bottom-left: each row is a (marker_color, fill_color) pair.
- Scattered L-shaped corner markers of each marker color define opposite corners of axis-aligned rectangles.
- Some markers are solid 2×2 blocks instead of corner pairs.

Rule:
- Read legend bottom-to-top; for each entry find all pixels of that marker color.
- Determine which two bbox corners carry L-shaped anchors — the *other* diagonal defines the fill region.
- Fill the diamond (all shortest-path cells between those two corners) with the legend's fill color.
- Solid 2×2 marker blocks are simply recolored to the fill color.
- Later legend entries are protected from earlier fills.

Key insight: The L-shaped anchors sit on the marker diagonal; the fill goes on the opposite diagonal of the bounding box.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="eval" puzzle_id="142ca369" lines="6-19">
<description>
Trace diagonal rays from triangular arrows, reflecting off colored wall pixels.

Input structure:
- bg grid with monochrome 3-pixel L-shaped "arrows" (2x2 bbox missing one corner) and scattered colored wall pixels.

Rule:
- Each arrow fires a ray diagonally away from its missing corner, painting bg cells in its color.
- A head-on wall hit reverses both axes; a side-touch wall hit flips the touched axis only.
- On any bounce the ray adopts the wall's color and recolors the contact cell.
- Ray stops when it exits the grid or reaches another arrow.

Key insight: Two distinct reflection types (corner vs side-touch) plus color inheritance at each bounce drive the path geometry.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="eval" puzzle_id="abc82100" lines="70-86">
<description>
Use local source-target-pattern examples to replace matching source pixels.

Input structure:
- Black background with isolated colored pixels and teal (8) pattern components.
- A rule example is a 4-neighbor colored pair source->target whose continued
  direction reaches an anchor touching a teal pattern component.

Rule:
- For each rule example, copy that teal component onto every other pixel of
  source.color, recoloring the copied component to target.color.
- Translation maps the rule anchor onto the source pixel being replaced.
- Remove teal components and all pixels used by rule examples.

Key insight: The two adjacent colored pixels define the color substitution and
the anchor point; the nearby teal component is the stencil that gets recolored.
</description>
</reference_puzzle>

<reference_puzzle index="5" source="eval" puzzle_id="a6f40cea" lines="74-88">
<description>
Complete partial paths inside a framed rectangle by projecting external color traces inward.

Input structure:
- bg, a 1px rectangular frame (border color) with uniform interior fill.
- Colored components outside the frame touching its edges.

Rule:
- Output = the frame's interior, filled with the fill color.
- Map each outside component's frame-touch points into interior coordinates.
- Monochrome components become solid Manhattan stripes between mapped endpoints.
- Two-color components become alternating-color Manhattan paths.

Key insight: Endpoint colors of two-color traces anchor the alternating phase deterministically.
</description>
</reference_puzzle>

<reference_puzzle index="6" source="eval" puzzle_id="8f3a5a89" lines="22-36">
<description>
Paint an inner border around the region reachable from a corner marker pixel.

Input structure:
- bg = most common color; wall color (blue) forms barriers dividing the grid.
- A unique single pixel (magenta) sits on one corner — the start marker.
- Some wall-colored pixels are structural (border-touching, enclosing regions); others are decorations.

Rule:
- Flood-fill from the marker corner to find all reachable bg cells, blocked by structural walls.
- Erase wall-colored decorations hidden behind structural walls (not reachable from marker side).
- Paint orange (color 7) on every reachable bg cell adjacent to a wall or grid edge — forming a 1-cell inner border.

Key insight: Only border-connected wall components act as real barriers; interior wall fragments are decorations to be removed.
</description>
</reference_puzzle>

<reference_puzzle index="7" source="eval" puzzle_id="5961cc34" lines="8-21">
<description>
Build a red ray network by activating blue blobs via directional green ports.

Input structure:
- bg = most common color. Large blue blobs with adjacent green pixels marking port directions.
- Start marker: yellow pixel atop a short red segment at grid bottom.

Rule:
- Extend a ray upward from start; on hitting a blue blob, recolor it red.
- Activated blobs emit thick rays through green ports until boundary or another blob.
- Cascade until no new blobs are reached.

Key insight: Green pixels encode ray direction and width; the network activates as a chain reaction from one start point.
</description>
</reference_puzzle>

<reference_puzzle index="8" source="eval" puzzle_id="53fb4810" lines="105-119">
<description>
Extend marker tiles outward from '+' shapes by repeating them to the grid edge.

Input structure:
- bg = most common color.
- A dominant non-bg color forms several '+'-shaped objects.
- Small colored patches (1–few cells) sit adjacent to each arm of a '+'.

Rule:
- For each marker tile touching a '+', determine which side of the '+' it sits on.
- Project the tile away from the '+' in that direction, repeating the tile pattern periodically until the grid boundary.
- '+' pixels are never overwritten.

Key insight: Each marker's attachment side determines its extension direction; the pattern repeats periodically, not as a single copy.
</description>
</reference_puzzle>

<reference_puzzle index="9" source="eval" puzzle_id="e12f9a14" lines="138-152">
<description>
Emit diagonal beams from framed 2x2 blocks through frame gaps; beams interact on collision.

Input structure:
- bg = most common color. Wall color (e.g. green) forms 4x4 frames around 2x2 monochrome blocks.
- Each frame has one or more gap pixels where the wall is missing.

Rule:
- Each gap emits a beam of the inner block's color, traveling diagonally outward (direction set by gap position relative to the block).
- Beams paint every cell they cross until they hit the grid edge or collide.
- 90° collision: both beams turn 45° toward their bisector direction; the intersection cell is left blank (1-pixel gap).
- 45° collision: beams merge into a zigzag (alternating cardinal + diagonal steps) with no gap at the meeting point.

Key insight: Two distinct collision rules — 90° produces a gap and single-pixel diagonal lines; 45° produces no gap and thicker zigzag bands.
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

<reference_puzzle index="11" source="eval" puzzle_id="31f7f899" lines="7-21">
<description>
Sort vertical bars by height and redistribute them across fixed colored slots.

Input structure:
- bg background; a horizontal base line of a single color spans the full grid width.
- Several monochrome vertical bars of distinct colors rise upward from the base line.

Rule:
- Rank bar shapes by height (shortest to tallest).
- Rank original x-positions (slots) left to right.
- Place the i-th shortest shape into the i-th leftmost slot, recoloring it to that slot's original color.
- The base line is preserved unchanged.

Key insight: Each slot keeps its x-position and color identity; only the bar silhouettes are redistributed among slots by height order.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="eval" puzzle_id="cb2d8a2c" lines="13-26">
<description>
Extend a green path from one grid edge to the opposite, routing around red bars at a distance dictated by their blue pixel count.

Input structure:
- bg = azure (8). Red/blue bars (color 2/1) act as obstacles.
- A single green (3) pixel sits on a grid border — the path start.

Rule:
- Each bar's keep-out radius equals its number of blue pixels; the path must stay farther than that from every bar cell.
- The green pixel extends inward toward the opposite edge, turning at right angles to navigate around forbidden zones.
- After the path is drawn, all blue pixels become red.

Key insight: Blue pixels on a bar encode its danger radius — more blue means a wider no-go zone the path must respect.
</description>
</reference_puzzle>

<reference_puzzle index="13" source="eval" puzzle_id="269e22fb" lines="46-60">
<description>
Complete a visible crop into the shared 20x20 binary motif in its matching orientation.

Input structure:
- The grid is a two-color rectangular crop of one rotation/reflection of the same 20x20 motif.
- One color is the motif foreground and the other color is the motif background.

Rule:
- Try each input color as the foreground mask.
- Match that mask against every possible window in every dihedral orientation of the motif.
- Repaint the uniquely matched full 20x20 motif with the matched foreground/background colors.

Key insight:
- The examples are partial windows of one canonical binary pattern; the local crop is distinctive enough to identify the full orientation and offset.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="eval" puzzle_id="800d221b" lines="7-22">
<description>
Recolor a path-network by adjacent seed regions.

Input structure:
- bg = most common color.
- Path color P forms a thin orthogonal network with a solid square hub (NxN).
- Seed colors appear in colored regions (non-bg, non-P) adjacent to path branches.

Rule:
- Infer P as the non-bg color whose largest component is most network-like.
- Find the largest solid k×k all-P square (the hub); exclude it from recoloring.
- Recolor each path branch to the seed-region color it touches most.
- Recolor the hub center pixel to the seed whose branches touch the hub most.

Key insight: The hub stays uncolored; branches inherit color from adjacent seed regions by adjacency count.
</description>
</reference_puzzle>

<reference_puzzle index="15" source="eval" puzzle_id="136b0064" lines="25-39">
<description>
Draw a polyline by decoding 3x3 glyph instructions read across a divided grid.

Input structure:
- Left of a full-height vertical divider: 3x3 colored glyphs on bg.
- Right of divider: a single gray (5) start pixel.

Rule:
- Remove the divider; shift the gray start into the left canvas.
- Read glyphs in column-major order (left→right, top→bottom).
- Each glyph's shape (pixel area) encodes a segment: area 4 → down 2, area 5 → right 3, area 6 → left 4, area 7 → left 2.
- Append segments sequentially, stepping down one cell between each.

Key insight: Glyph pixel count (area) uniquely determines both segment length and direction — the shape itself is the instruction.
</description>
</reference_puzzle>

<reference_puzzle index="16" source="eval" puzzle_id="4c7dc4dd" lines="7-21">
<description>
Learn a transformation rule from a training pair of framed sprites, then apply it to a test sprite.

Input structure:
- Grid contains several colored square frames (hollow borders of a single color).
- Each frame's interior holds a small pattern sprite; one frame's interior is blank (all black).
- Two frames form a training pair (their interiors share the same color palette); one remaining frame is the test case.

Rule:
- Identify the training pair by matching interior color sets; the other non-blank frame is the test input.
- Infer the rule mapping source→destination (e.g. reflection, color inversion, gap filling, endpoint extraction, or checkerboard complement).
- Apply that same rule to the test sprite; output the transformed interior.

Key insight: The puzzle is an analogy — "A is to B as C is to ?" — where you must discover the visual rule from one example pair and generalize it.
</description>
</reference_puzzle>

<reference_puzzle index="17" source="eval" puzzle_id="b10624e5" lines="24-39">
<description>
Replicate a motif across quadrants with scaling and reflection keyed to red anchor blocks.

Input structure:
- Grid divided into four quadrants by a full-row and full-column divider of a single non-bg color.
- Each quadrant contains a red (2) square anchor block of varying size.
- The top-left quadrant also has a colored motif attached to its red anchor.

Rule:
- Use the top-left quadrant's motif as the source pattern.
- For each other quadrant, scale the motif by the ratio of that quadrant's red block size to the source red block size.
- Reflect horizontally for right-side quadrants, vertically for bottom quadrants (both for bottom-right).
- Align the scaled/reflected motif so its red region overlaps the target anchor, then stamp non-bg pixels.

Key insight: Red block size encodes the scale factor; quadrant position encodes the reflection axes.
</description>
</reference_puzzle>

<reference_puzzle index="18" source="eval" puzzle_id="898e7135" lines="7-20">
<description>
Inflate a frame's hole-layout and fill each hole with its matching piece.

Input structure:
- bg = most common color. Largest component is a monochrome frame enclosing bg holes.
- Outside pieces composed of k×k blocks; k = side of smallest solid square piece.

Rule:
- Extract connected hole regions inside the frame as slots.
- Match each piece to a slot by comparing block-level shapes under rotation/flip.
- Output the layout scaled by k, with pieces placed in slots; frame color as background.

Key insight: Pieces are k×k-block-scaled versions of holes; downsampling reveals shape matches under all 8 orientations.
</description>
</reference_puzzle>

<reference_puzzle index="19" source="eval" puzzle_id="71e489b6" lines="7-19">
<description>
Highlight black defects in blue foreground by surrounding them with orange.

Input structure:
- Blue (1) foreground on black (0) bg; small black holes or protrusions within blue regions; occasional thin blue spurs into black areas.

Rule:
- Thin blue pixels mostly surrounded by black are absorbed into the background.
- Small black components (≤3 cells) and leaf pixels of larger black components are "defects."
- Each defect gets an orange (7) 8-neighbor halo; the defect pixels themselves stay black.

Key insight: Only structural anomalies—tiny isolated patches and dangling endpoints—are marked, not bulk black regions.
</description>
</reference_puzzle>

<reference_puzzle index="20" source="eval" puzzle_id="9aaea919" lines="38-51">
<description>
Transfer sprites from red-marked lanes to green-marked lanes, recoloring and stacking.

Input structure:
- bg = most common color. Red (2) and green (3) thin bars define vertical lanes.
- Each lane contains repeated identical monochrome sprites in an evenly spaced vertical grid.

Rule:
- Each red lane's sprites turn grey (5); their count transfers to the nearest green lane.
- New sprites fill empty lattice slots bottom-up, using the source shape recolored to the destination lane's sprite color.
- All marker bars are removed.

Key insight: Sprite count is conserved across lanes — red sprites become grey, and equally many new sprites stack into the green lane's regular grid.
</description>
</reference_puzzle>

你的任务：
以参考题为标杆，运用发散性思维进行创新扩展。请发挥创造力，构思全新且巧妙的推理谜题。用中文创造全新的 ARC 谜题描述。
创新方式包括但不限于：
**机制重组**：将多道参考题各自的核心机制提取出来，组合成一个新的多步管道，用到新的场景。
**机制变异**：取几道参考题的核心机制，以机制为基础进行创新并与其他多种机制进行创新重组，使解题路径发生质变而非量变。
**逆向构造**：取一道参考题的输出模式，思考"什么样的输入规则能产生类似的输出结构但需要完全不同的推理过程"。

严格使用下面的 XML 风格格式输出：

<new_puzzle>
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。
</new_puzzle>

请一次性生成 5 个 `<new_puzzle>`，若有较多合适的创新，也可以生成更多，不设上限，但不要输出额外解释。

注意事项：
1. 任务不应依赖大量先验知识或复杂计算，而应依赖对模式、结构与空间关系的直觉推理。
2. 关于"计算"的边界：
   - **允许使用**（这些都是人类能心算的视觉量）：数像素 / 孔洞 / 对象 / 邻居（≤10 量级）、比大小、按某个可见属性排序、奇偶判断、取最大 / 最小、用一个计数值作为另一个等量几何量（如 N 个像素 → 长度 N 的射线、面积 N → 边长 N）、判断接触 / 相邻 / 包含、粗略中心位置（簇的视觉中点）、"距离"作为"几步"而不是公式等人类能够直观感受计算的内容。
   - **禁止使用**（这些是算法量而非视觉量，会迫使解题者依赖公式而非视觉直觉）：曼哈顿 / 欧氏 / Chebyshev 距离作为精确查表键、凸包面积、二进制位编码 / 位运算、模运算（取模） / 余数、拓扑距离（BFS / 最短路 / 跳数）、精确几何质心坐标、凸角 / 凹角计数等复杂且不易被看出来的计算都是禁止使用的。
3. 问题必须能从少量示例中推断出来（通常为 3 对训练样例 + 1 个测试样例）。设计时从人类解题者的视觉角度思考：规则涉及的所有关键要素（对象、关系、属性）必须在网格上肉眼可见，不应依赖人眼无法直接观察到的隐含信息（例如颜色id和颜色亮度是禁止使用的）或需要推算的隐藏属性。规则背后的逻辑本身可以有一定挑战性，但解题所需的所有线索必须能从网格中直接获取。
4. 生成的问题描述旨在促进后续用于创建题目网格的代码生成。因此，对输入 / 输出网格的描述以及规则总结必须极其详细；语言风格应易于理解；对于特定对象（如箭头），必须在描述中明确其绘制方式。
5. 描述谜题时避免使用 0 或 1 这类颜色编号；应使用颜色名称，例如 "red"。可用颜色如下：Black、Blue、Red、Green、Yellow、Grey、Magenta、Orange、Azure、Maroon。
6. 5 个新题的核心推理机制应明显不同——即从观察样例到归纳规则的认知路径不同，而不仅仅是场景、对象或颜色不同。
7. 题目难度应来自机制的隐蔽性而非步骤数。用参考题作为难度标尺——如果你生成的题一个样例就能猜到规则，说明机制太透明，需要加深隐蔽性。
8. 优先从参考题的机制出发进行重组或推演，而非从物理隐喻出发。物理隐喻容易导致大量题目归结为一种模式，例如"沿方向移动到阻挡处停止"。
9. 题目的难度优先来自规则的**结构层数**，而不是单一步骤的复杂度。关于"结构层数"的判定标准：**只有解题者需要主动推理才能完成的步骤才算一层**。肉眼一扫即可获得的信息（如"找到缺口在哪一侧"、"数出像素个数"、"识别哪个对象最大"）不构成独立的推理层——它们只是变换的前置输入，不增加认知难度。真正的多层结构是指每一层都需要解题者发现一个非显然的规律。以两层结构为例：
   - ✗ 伪两层："找到缺口方向"→"把东西推到对面" — 第一步肉眼可见，实际只有一层推理。
   - ✓ 真两层："通过孔洞数量匹配颜色图例"→"用匹配到的颜色重绘对象" — 两步都需要非平凡的归纳。

   如果你发现自己写出的题只是"对每个对象，根据它的某个属性做某变换"的单步规则，请引入一个需要主动推理的前置步骤。**用堆叠 tie-break 规则或复杂数学公式来"加难度"是劣解；用多一层需要归纳的结构嵌套来"加难度"才是好解**。
10. 在符合ARC题目和提示词要求的基础上，鼓励尝试创新规则为多层结构的题目，不局限于两层，可以创新三层或四层结构的题目，但不可为了多结构层数违背要求。

生成后对生成的题执行以下四项审查（审查内容不需要保存到文件）：
1. 原创性
  新题不可与上述参考题相同或类似，不可只是替换颜色、方向、对象类型等表层要素。判断标准：新题的核心推理路径应与任何一道参考题不同。
2. 确定性
  规则完全确定，同一输入只能产生唯一输出，每一步操作都有明确定义。
3. 可实现性
  所有必要细节都已给出——对象如何生成、如何识别、如何变换、边界情况如何处理，读完描述后可以直接写代码。
4. ARC 合规性
  网格尺寸合理（≤30×30），使用颜色不超过 10 种，规则复杂度适中，人类能从 3-5 个样例中发现规律。没有使用上述注意事项第二条禁止使用的复杂计算。


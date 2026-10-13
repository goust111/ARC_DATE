你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="eval" puzzle_id="4a21e3da" lines="8-20">
<description>
Cut a sprite along border-control rays and slide pieces to grid corners.

Input structure:
- bg, fg (main object, ~centered), ctrl (1-2 pixels on grid edges).

Rule:
- Draw rays inward from each ctrl pixel along its row/col, painting only bg cells up to the farthest fg pixel (intermittent).
- Rays partition the grid; fg on the cut line stays in place.
- Cut pieces in regions touching a control edge slide to that corner; others vanish.

Key insight: Only regions sharing a border edge with a control pixel keep their piece; the ray skips fg cells.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="eval" puzzle_id="409aa875" lines="68-81">
<description>
Fire projectiles from 3-pixel triangle sprites; mark collisions with maroon or blue.

Input structure:
- bg = most common color. Non-bg 3-pixel sprites shaped as axis-aligned L's or diagonal V's.

Rule:
- Each triangle has an apex and an emission direction determined by its shape: L-shapes fire diagonally away from the missing corner; V-shapes fire along the axis away from the open side.
- A projectile lands exactly 5 cells from the apex in the emission direction.
- If it hits an existing sprite, that entire sprite is recolored to maroon (9).
- If two projectiles land on the same empty cell, that cell becomes blue (1); a lone projectile on empty bg paints maroon (9).

Key insight: The triangle's orientation encodes both direction and target; dual-projectile collisions on open bg produce blue, not maroon.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="eval" puzzle_id="898e7135" lines="7-20">
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

<reference_puzzle index="4" source="train" puzzle_id="6350f1f4" lines="62-81">
<description>
Decode a block-grid image by swapping each sprite-coded cell with a solid cell and each blank cell with the sprite.

Input structure
The grid is partitioned by black separator rows and columns into equal-size
square blocks. Each block is either a copy of one two-color sprite or a
uniform blank block of the sprite's minority color; grey pixels are erased
cells that act as wildcards.

Rule
Infer the repeated sprite from fully visible mixed-color blocks. Replace a
block matching that sprite, ignoring grey erasures, with a solid block of
the sprite's majority color. Replace a uniform blank block with the sprite.
Separator rows and columns, including grey erasures on them, become black.

Key insight
The small sprite is a block-level code for one macro-cell state, and the
uniform minority-color block is the other state; the output renders the
opposite code in each macro-cell while cleaning erased pixels.
</description>
</reference_puzzle>

<reference_puzzle index="5" source="eval" puzzle_id="446ef5d2" lines="8-20">
<description>
Assemble scattered rectangular pieces into a rectangle at an L-marker corner.

Input structure:
- bg = most common color. Non-bg rectangular pieces scattered on the grid.
- 3-pixel L-marker (2x2 minus one cell) tags the corner; the gap is the corner cell.

Rule:
- rect_col = most common color across all piece pixels; all outer rectangle edges must be rect_col.
- Translate pieces (no rotation) to tile the rectangle; touching edges must match, with rect_col as wildcard.

Key insight: rect_col acts as wildcard during edge matching — only non-rect_col pixels at shared boundaries must agree.
</description>
</reference_puzzle>

<reference_puzzle index="6" source="train" puzzle_id="182e5d0f" lines="89-103">
<description>
Move each gray marker along its green path until it reaches the cell next to a black-green-black gate, erasing the path behind it.

Input structure:
- The background is the dominant color.
- Foreground components are made from a path color, a paired gate-end color, and single marker pixels.
- A gate is a straight end-center-end motif whose center has the path color and whose two ends share the gate-end color.

Rule:
- In every 4-connected foreground component with exactly one marker and one gate, erase the marker and all path-color pixels except the gate center.
- Place the marker on the unique path-color pixel adjacent to the gate center but not between the two gate ends.
- Components without both a marker and a gate stay unchanged.

Key insight: The black-green-black motif marks the stopping gate; the gray marker consumes the connected green path until it rests on the path cell immediately beside that gate.
</description>
</reference_puzzle>

<reference_puzzle index="7" source="eval" puzzle_id="1ae2feb7" lines="30-45">
<description>
Extend row bars periodically across the opposite side of a separator column.

Input structure:
- bg = background. A near-solid vertical separator divides the grid into two sides.
- One side contains horizontal colored bars; the other side is blank target space.
- Bars may be stacked by row and may sit at different distances from the separator.

Rule:
- Identify the separator as the column whose non-bg cells are one color.
- Treat the denser side as the bar side and collect each contiguous row run.
- Process bars from nearest to farthest from the separator; for each bar, paint its
  color every bar-length cells into the blank side without overwriting earlier paint.

Key insight: Each bar's length is the period for its extension, and nearer bars claim target cells first.
</description>
</reference_puzzle>

<reference_puzzle index="8" source="train" puzzle_id="3ad05f52" lines="128-142">
<description>
Paint the maze corridor and framed rooms lying on shortest paths between the two colored endpoint components.

Input structure
- The grid has a dominant background, one wall color forming thin orthogonal frames, and one endpoint/fill color.
- The endpoint/fill color forms exactly two 4-connected solid rectangular components; their larger side lengths define the framed room sizes.

Rule
- Treat wall pixels as blocked and every other pixel as passable.
- Find every passable pixel that lies on a shortest 4-connected path between the two endpoint components.
- Also fill any framed square room of an endpoint-derived size that intersects that shortest-path mask.

Key insight
- Shortest paths identify the route through open connector corridors, while the wall-bordered square rooms need to be expanded from the route centerline to their full interior.
</description>
</reference_puzzle>

<reference_puzzle index="9" source="train" puzzle_id="7ec998c9" lines="14-28">
<description>
Draw a blue folded corner-to-corner route through the lone marker's column.

Input structure:
- A rectangular grid filled with one background color.
- Exactly one non-background marker pixel; blue is absent from the input.

Rule:
- Preserve the marker and repaint background pixels in its column blue from top to bottom.
- If the marker is in the unique center column, connect that column to the top-right and bottom-left corners.
- Otherwise, connect the column to the top-left and bottom-right corners.

Key insight:
The marker's column is the route spine; its central vs off-center placement selects which opposite corner pair gets the horizontal edge segments.
</description>
</reference_puzzle>

<reference_puzzle index="10" source="eval" puzzle_id="db695cfb" lines="8-19">
<description>
Connect same-color diagonal pixel pairs, then emit perpendicular rays from differently-colored pixels on those diagonals.

Input structure:
- bg = most common color. Line color (e.g. blue) has pixel pairs aligned on shared diagonals. Trigger color (e.g. magenta) pixels sit along those diagonals.

Rule:
- Draw diagonal lines connecting each pair of same-color aligned pixels.
- Where the diagonal crosses a trigger pixel, emit perpendicular diagonal rays (in the trigger's color) extending to the grid boundary.

Key insight: Trigger pixels act as beam-splitters — the perpendicular rays rotate 90° from the connecting diagonal's orientation.
</description>
</reference_puzzle>

<reference_puzzle index="11" source="eval" puzzle_id="7b0280bc" lines="54-69">
<description>
Recolor the shortest path between two terminal squares in a square-and-track graph.

Input structure:
- bg = most common color.
- Two non-bg colors form solid NxN squares (N>=2): one color marks exactly 2 terminals, the other marks stations.
- A third non-bg color forms 1-cell-wide track segments connecting pairs of squares (graph edges).

Rule:
- Find the shortest path (fewest edges) between the two terminal squares.
- Recolor every station square on that path to green.
- Recolor every track segment along that path to grey.
- Everything else unchanged.

Key insight: The grid encodes a graph where squares are nodes and tracks are edges; the task is graph shortest-path, not spatial proximity.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="eval" puzzle_id="6e453dd6" lines="39-52">
<description>
Slide black shapes rightward to a grey separator, then shoot red rays from enclosed holes.

Input structure:
- A vertical grey (5) separator column divides the grid into left and right halves.
- Black shapes sit on the left side against a background color (bg).

Rule:
- Translate each black connected component rightward until its right edge is flush with the separator.
- After sliding, find enclosed bg regions (holes) within the black shapes that touch the column just left of the separator.
- For each row spanned by such a hole, emit a red ray across the entire right half of the grid.

Key insight: Only holes adjacent to the separator trigger rays — the sliding creates enclosures that act as "nozzles" projecting rightward.
</description>
</reference_puzzle>

<reference_puzzle index="13" source="train" puzzle_id="1e5d6875" lines="26-45">
<description>
Paint a recolored diagonal copy from each L triomino source object.

Input structure:
- The background is the dominant color.
- The only foreground objects are 4-connected red and grey L triominoes,
  each occupying three pixels of a 2x2 box.

Rule:
- Grey L objects paint a yellow copy translated one diagonal step toward
  their missing 2x2 corner.
- Red L objects paint a green copy translated one diagonal step away from
  their missing 2x2 corner.
- Copies are clipped by the grid boundary; red-derived green copies are
  painted after grey-derived yellow copies where the projections collide.

Key insight:
- The missing corner of each 2x2 L object identifies its diagonal axis, and
  the source color determines which side of that axis receives the copy.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="train" puzzle_id="8e301a54" lines="8-23">
<description>
Move every foreground object straight downward by a number of rows equal to its own pixel count.

Input structure:
- The grid has one dominant background color.
- The foreground is made of separated 4-connected monochromatic objects.
- Objects may move partly below the bottom edge of the grid.

Rule:
- Clear the canvas to the background.
- For each foreground object, compute its area in pixels.
- Redraw the object at the same columns and that many rows lower, clipping pixels below the grid.

Key insight:
- The downward displacement is the object's own area; color and shape only identify which pixels move together.
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

<reference_puzzle index="16" source="train" puzzle_id="68bc2e87" lines="46-60">
<description>
Recover the bottom-to-top stacking order of overlaid rectangular color frames.

Input structure:
- The background is one color, and every foreground color is a one-cell-thick rectangular frame.
- Overlapping frame pixels show only the color of the frame that is highest at that location.

Rule:
- Reconstruct each frame as the border of the bounding box of its visible pixels.
- At every cell covered by multiple reconstructed frames, the visible frame color is above the other frames covering that cell.
- Return the unique bottom-to-top order of frame colors as a single-column grid.

Key insight:
The hidden parts of each frame do not need to be seen directly; the visible pixels reveal the full frame bbox, and overlap cells reveal the layer constraints.
</description>
</reference_puzzle>

<reference_puzzle index="17" source="train" puzzle_id="bd283c4a" lines="7-23">
<description>
Pack all pixels into color-frequency runs along bottom-up columns.

Input structure
The input is a rectangular grid whose colors are all foreground material;
only the number of pixels of each color matters. In the observed task
instances, every present color has a distinct pixel count.

Rule
Count the pixels of each color and order the colors from most frequent to
least frequent. Rewrite a same-size grid by placing each color's full count
as one contiguous run while scanning columns from left to right and, within
each column, from bottom to top.

Key insight
The output preserves the multiset of colors but discards all original
positions, turning the grid into a frequency-sorted column packing.
</description>
</reference_puzzle>

<reference_puzzle index="18" source="train" puzzle_id="8dab14c2" lines="120-133">
<description>
Complete each boundary tab or notch by adding the complementary feature on the opposite side of the foreground object.

Input structure
- The grid has exactly two colors: a dominant background and one foreground object.
- The foreground object is a thick rectilinear shape with one-pixel tabs and notches on its boundary.

Rule
- A foreground tab is a one-neighbor boundary pixel; project through its full row or column span and remove the opposite edge pixel to make a matching notch.
- A background notch is a one-pixel cut with three foreground 4-neighbors; project across the full row or column span and add a tab just outside the opposite edge.
- Notches immediately beside a tab source are not independent sources, because that gap belongs to the tab's local boundary shape.

Key insight
- The changed pixels are not color-specific noise; they are paired jigsaw-like boundary features on opposite sides of the same thick row or column cross-section.
</description>
</reference_puzzle>

<reference_puzzle index="19" source="train" puzzle_id="bd5af378" lines="25-42">
<description>
Rewrite a corner L block into two nested L borders and a diagonal trace.

Input structure
A rectangular grid has exactly two colors: one color forms a one-cell-thick
L on two adjacent outer edges, and the other color fills the remaining
rectangle.

Rule
Normalize the L to the top-left corner. Paint the outer top-left L with the
fill color, the next inner L with the original border color, and the main
diagonal of the remaining inner rectangle with the fill color. All other
cells become color 8, then the original orientation is restored.

Key insight
The corner of the input L determines the orientation; after flipping to a
canonical top-left L, the output is the same nested-layer pattern for every
example.
</description>
</reference_puzzle>

<reference_puzzle index="20" source="eval" puzzle_id="8e5c0c38" lines="9-21">
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
   - **禁止使用**（这些是算法量而非视觉量，会迫使解题者依赖公式而非视觉直觉）：曼哈顿 / 欧氏 / Chebyshev 距离作为精确查表键、凸包面积、二进制位编码 / 位运算、模运算 / 余数、拓扑距离（BFS / 最短路 / 跳数）、精确几何质心坐标、凸角 / 凹角计数等复杂且不易被看出来的计算都是禁止使用的。
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


你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="eval" puzzle_id="247ef758" lines="35-46">
<description>
Move colored sprites from the left strip into a framed area at marker-defined targets.

Input structure:
- bg background; rectangular frame on right with colored markers on its edges; sprites in a narrow left strip separated by a constant-color divider column.

Rule:
- For each color, target rows/cols are where both opposite frame edges share that color. Targets are the Cartesian product.
- Erase each sprite from the left and stamp it centered at every target. Higher sprites cover lower ones.

Key insight: One sprite replicates to a grid of row × column marker intersections.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="eval" puzzle_id="db695cfb" lines="8-19">
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

<reference_puzzle index="3" source="eval" puzzle_id="7b80bb43" lines="8-21">
<description>
Remove diagonal staircase noise and complete broken straight lines.

Input structure:
- bg = background; one foreground color fg forming axis-aligned lines.
- Some fg pixels form 45° diagonal chains (8-connected but 4-disconnected from lines).

Rule:
- Delete all diagonal-only fg pixels (those with no orthogonal fg neighbor, only diagonal ones).
- Fill short gaps in horizontal/vertical lines where a deleted diagonal was nearby.
- Prune remaining diagonal-adjacent leaf pixels near former diagonal sites.

Key insight: Diagonal pixels are hints marking where a straight line is broken; their span sets the maximum gap size to fill.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="eval" puzzle_id="3dc255db" lines="37-53">
<description>
Eject content pixels from cup-shaped frames outward along each cup's tail direction.

Input structure:
- bg = background. Hollow cup/Y/funnel frames, each a unique border color.
- Each cup has a convergent tail spike on one side and an open mouth on the opposite side.
- Content pixels (a single non-border color per cup) sit inside the cup's mouth interior.

Rule:
- For each cup, identify its tail spike and tail direction.
- Erase all content pixels from the mouth interior.
- Draw a straight line of content-colored pixels from the spike outward in the tail direction,
  one pixel per erased content pixel; cells beyond the grid edge are dropped.

Key insight: The tail spike is the unique convergent point whose outward direction is opposite
the mouth opening where the content lives.
</description>
</reference_puzzle>

<reference_puzzle index="5" source="eval" puzzle_id="78332cb0" lines="10-23">
<description>
Reorder 5×5 blocks into a connected strip by matching edge foreground pixels.

Input structure:
- bg = background; sep = divider color (full row/column lines).
- Several 5×5 sprite blocks in a grid, separated by sep lines.

Rule:
- Two blocks "connect" when their facing edges share foreground pixels at the same offset (e.g., A's right edge ∩ B's left edge).
- Chain all blocks in the orientation (horizontal or vertical) that admits a complete sequence.
- Output is a 1×N strip of blocks in chain order, separated by sep lines.

Key insight: Edge-contact overlap determines both block ordering and output orientation.
</description>
</reference_puzzle>

<reference_puzzle index="6" source="eval" puzzle_id="8e5c0c38" lines="9-21">
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

<reference_puzzle index="7" source="eval" puzzle_id="80a900e0" lines="69-82">
<description>
Extend diagonal rays outward from outer-border segments adjacent to diamond shapes on a checkerboard.

Input structure:
- Checkerboard bg (two most common colors).
- Hollow diamond outlines (8-pixel) in a non-bg color (inner border).
- Short diagonal line segments in another non-bg color adjacent to each diamond edge (outer border).

Rule:
- For each outer-border segment, determine the outward direction (perpendicular to the segment, away from the diamond).
- From each endpoint of the segment, cast a diagonal ray outward, painting over bg pixels with the segment's color until hitting a non-bg pixel or the grid edge.

Key insight: Rays emit only from the two endpoints of each outer-border segment, not every pixel, and travel diagonally away from the diamond.
</description>
</reference_puzzle>

<reference_puzzle index="8" source="eval" puzzle_id="71e489b6" lines="7-19">
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

<reference_puzzle index="9" source="eval" puzzle_id="7b5033c1" lines="7-21">
<description>
Straighten a colored snake path into a single-column strip preserving pixel order.

Input structure:
- bg = most common color.
- Non-bg pixels form a single 4-connected path (snake) with exactly two endpoints.
- Each path pixel may be a different non-bg color.

Rule:
- Find the two endpoints (cells with only one path neighbor) of the snake.
- Begin at the top-left-most endpoint and trace the path cell by cell.
- Output an (N×1) column listing each cell's color in traversal order.

Key insight: The path is a simple chain with no branches; choosing the top-left-most endpoint fixes a unique reading direction.
</description>
</reference_puzzle>

<reference_puzzle index="10" source="eval" puzzle_id="409aa875" lines="68-81">
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

<reference_puzzle index="11" source="eval" puzzle_id="7c66cb00" lines="9-24">
<description>
Dock two-tone sprites into color-matched horizontal bars with a color swap.

Input structure:
- bg = most common color.
- Top area: small two-tone sprites (outer color + inner color).
- Below: horizontal bars spanning most of the row, each with a fill color and border color on edges.

Rule:
- Match each sprite to the bar whose fill color equals the sprite's outer color.
- Place the sprite at its original x-position, aligned to the bar's bottom.
- Recolor: sprite outer → bar border, sprite inner → bar fill.
- Erase original sprites from the top area.

Key insight: Color matching is inverted — the sprite's dominant color identifies its target bar, then gets swapped to the bar's accent color.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="eval" puzzle_id="135a2760" lines="6-19">
<description>
Repair noisy periodic patterns inside frame-delimited grid regions.

Input structure:
- Frame color (most common) forms full rows/cols, dividing the grid into rectangular regions.
- bg (second most common) fills empty space; each region contains a small colored tile repeated periodically with a few corrupted pixels.

Rule:
- For each region, find the bounding box of non-bg content.
- Infer the smallest repeating tile that explains the pattern (majority vote per residue position), requiring it repeat at least 2x per axis.
- Replace the region with the clean tiled version.

Key insight: Tile dimensions are unknown and must be inferred; prefer the smallest tile with genuine repetition over one that memorizes the entire region.
</description>
</reference_puzzle>

<reference_puzzle index="13" source="eval" puzzle_id="45a5af55" lines="5-18">
<description>
Build nested square frames from a stack of horizontal color stripes.

Input structure:
- Full-width horizontal stripes of distinct non-bg colors, stacked vertically.
- Each stripe has uniform color and variable height (1-4 rows).

Rule:
- Read stripes bottom to top.
- Bottom stripe becomes the center solid square with side length = stripe height.
- Each subsequent stripe wraps the current shape with a square frame of thickness = stripe height, in that stripe's color.

Key insight: Stripe height doubles as both frame thickness and (for the innermost) center-square side length; output dimensions are fully determined by the stripe heights.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="eval" puzzle_id="a6f40cea" lines="74-88">
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

<reference_puzzle index="15" source="eval" puzzle_id="89565ca0" lines="120-136">
<description>
Convert each colored partitioned frame into a horizontal bar whose length is its number of enclosed regions.

Input structure:
- Black background with several axis-aligned colored line drawings.
- The line drawings are damaged by scattered pixels of one non-background noise color and by overlaps with other drawings.
- Each real drawing is a rectangular partitioned frame enclosing one or more background regions.

Rule:
- For each non-background color, reconstruct its horizontal and vertical wall segments from dense row/column evidence.
- Count the 4-connected background regions fully enclosed by those reconstructed walls.
- The color with no enclosed region is the noise/pad color.
- Sort drawing colors by enclosed-region count; equal counts are ordered by which reconstructed frame covers the other.

Key insight:
- The scattered color is not an object; it only pads the histogram, while each true object's room count is stable after rectilinear wall reconstruction.
</description>
</reference_puzzle>

<reference_puzzle index="16" source="eval" puzzle_id="898e7135" lines="7-20">
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

<reference_puzzle index="17" source="eval" puzzle_id="28a6681f" lines="8-23">
<description>
Simulate water pixels falling under gravity into basins formed by colored obstacles.

Input structure:
- bg = 0. Water pixels = color 1 (blue).
- Other colors form static solid structures (walls, steps, containers).
- Water starts in unstable positions above or beside basins.

Rule:
- Each water pixel falls straight down through bg cells.
- When blocked below, it slides diagonally through the unique open side cell.
- When resting on a surface, it slides horizontally toward the nearest gap where it can fall again.
- Process repeats until no water pixel moves.

Key insight: Water seeks the deepest reachable basin via diagonal and horizontal sliding, not just vertical fall.
</description>
</reference_puzzle>

<reference_puzzle index="18" source="eval" puzzle_id="67e490f4" lines="8-22">
<description>
Fill holes inside a large template frame using shape-matched outside sprites.

Input structure:
- bg = most common color. One large monochrome frame object (template).
- Inside the frame: holes of various shapes (non-frame-color regions).
- Outside the frame: small monochrome sprites scattered on bg.

Rule:
- Crop to the frame's bounding box; fill entirely with frame color.
- For each hole, find an outside sprite whose shape matches (rotation/flip allowed).
- Fill the hole with the matched sprite's color. If multiple sprites match, use the most frequent color.

Key insight: Sprites act as shape-coded color keys — their shape identifies which hole they fill, their color determines the fill color.
</description>
</reference_puzzle>

<reference_puzzle index="19" source="eval" puzzle_id="581f7754" lines="8-22">
<description>
Slide multi-color sprites to align their marker pixels with matching isolated border anchors.

Input structure:
- bg = most common color.
- Multi-color sprites in the interior; each has a minority-color "marker" pixel.
- Isolated border pixels (non-bg with all 4-neighbors bg) act as anchors.

Rule:
- Match each sprite to the border anchor sharing its marker color.
- Top/bottom-edge anchor (non-corner): slide sprite horizontally so marker column = anchor column.
- Left/right-edge anchor (or corner): slide sprite vertically so marker row = anchor row.

Key insight: The anchor's edge position determines the slide axis — top/bottom means horizontal, left/right (or corner) means vertical.
</description>
</reference_puzzle>

<reference_puzzle index="20" source="eval" puzzle_id="142ca369" lines="6-19">
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


你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="eval" puzzle_id="3e6067c3" lines="7-20">
<description>
Connect framed squares via colored bridges following a control-pixel sequence.

Input structure:
- bg = background. Border color forms equal-sized framed squares with unique inner fill colors.
- Control row below squares: colored pixels read left-to-right encode visitation order.

Rule:
- Start at the square matching first control color c0.
- For each consecutive pair (ci, ci+1), find the ci+1-colored square aligned in the same row or column band, preferring nearest.
- Draw a bridge colored ci between the pair, spanning the gap with inner-square thickness.

Key insight: Row/column-band alignment plus proximity resolves ambiguity when inner colors repeat.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="eval" puzzle_id="20270e3b" lines="7-21">
<description>
Connect two sprites using matching connector pixels, then tight-crop.

Input structure:
- Background color `bg`, sprite color `obj`, connector color `ctrl` (small, 2 components).
- Two main foreground sprites, plus possibly tiny detached pixels belonging to one sprite.

Rule:
- Identify the **base** sprite (closest to grid top-left).
- Translate the other sprite so its connector sits one cell above the base's connector.
- Stamp the plug onto the base; remove connectors (recolor `ctrl` to `obj`).
- Output is the tight crop around all non-background pixels.

Key insight: The connector pixels define the unique docking point and direction.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="eval" puzzle_id="d35bdbdc" lines="120-134">
<description>
Transfer center colors between bordered sprites matched by color, keeping only path-connected ones.

Input structure:
- bg, grey (5) path network, and several small "bordered sprites" each having a border color and a single center pixel of a different color.
- Grey path forms a one-stroke trail under 8-connectivity; its start/end cells touch exactly two destination sprites via ports.

Rule:
- Destination sprites are determined by the grey trail's two endpoints.
- For each destination, find the source sprite whose border color equals the destination's center color.
- Replace the destination's center pixel with the source's center color.
- Delete all non-destination sprites; grey paths and destinations remain.

Key insight: Center color of each destination acts as a lookup key into the border colors of other sprites, enabling indirect color transfer.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="eval" puzzle_id="4a21e3da" lines="8-20">
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

<reference_puzzle index="5" source="eval" puzzle_id="3a25b0d8" lines="9-22">
<description>
Fill holes in a target object with colors from a source object's subsprites, then tight-crop.

Input structure:
- bg = background. Two similar 8-connected objects share an outline color.
- Source has colored subsprites in cavities; target has empty cavities.

Rule:
- Find the two 8-connected objects.
- Target = object with more hole area. Align subsprites to holes (rotation/flip if needed).
- Fill each hole with its matched subsprite color; tight-crop.

Key insight: Hole centers in the target are matched to source subsprite centers under the best rotation/flip transform.
</description>
</reference_puzzle>

<reference_puzzle index="6" source="eval" puzzle_id="6e4f6532" lines="26-41">
<description>
Dock train sprites onto station groups by matching coupler count, then orient via divider-color markers.

Input structure:
- bg; colored stripes (edge + internal) partition grid into rectangular regions.
- Color 8 forms train bodies; color 9 marks both isolated stations and train-attached couplers.
- Stripe-colored pixels on trains are orientation markers.

Rule:
- Stations = 8-connected 9-clusters not adjacent to any 8. Trains = components containing 8.
- Match train to station group where |couplers| == |station size|.
- Rotate/flip train so marker pixels face the same-color boundary stripe of the target region.
- Translate so couplers align exactly onto station cells; erase original position.

Key insight: Coupler count selects the docking target; marker color encodes which boundary stripe to face, fixing the unique orientation.
</description>
</reference_puzzle>

<reference_puzzle index="7" source="eval" puzzle_id="f931b4a8" lines="10-27">
<description>
Count the two upper quadrant control regions to size an output made from the lower quadrant pattern.

Input structure:
- The grid is an even square split into four equal quadrants.
- Black is the background.
- The upper-left and upper-right quadrants are control regions.
- The lower-left quadrant is a color source tile; the lower-right quadrant is either a stencil tile or empty.

Rule:
- Output height is the number of non-background pixels in the upper-left quadrant.
- Output width is the number of non-background pixels in the upper-right quadrant.
- If the lower-right stencil has foreground pixels, tile it over the output; each stencil background pixel is filled from the lower-left quadrant cell at the corresponding coarse tile position.
- If the lower-right stencil is empty, tile the lower-left quadrant directly over the output.

Key insight:
- The top quadrants specify only dimensions, while the bottom quadrants specify a two-level repeated pattern: a fine stencil and coarse fill colors.
</description>
</reference_puzzle>

<reference_puzzle index="8" source="eval" puzzle_id="78332cb0" lines="10-23">
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

<reference_puzzle index="9" source="eval" puzzle_id="136b0064" lines="25-39">
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

<reference_puzzle index="10" source="eval" puzzle_id="faa9f03d" lines="36-52">
<description>
Repair a one-cell-thick colored line network by replacing singleton markers, filling short orthogonal gaps, and redirecting yellow-marked arms to the edge.

Input structure:
- A rectangular grid with a dominant background.
- Line colors form one-cell-thick horizontal/vertical stroke fragments.
- Marker colors are exactly those whose 4-connected components are all singleton pixels.
- Red singleton markers are missing line cells; yellow singleton markers are directional controls next to a line endpoint.

Rule:
- Assign every marker to the nearest visible line color in row/column line-of-sight, preferring an endpoint supported along that same axis.
- For each line color, fill the gap between consecutive facing endpoints in a row or column.
- A yellow control erases the old straight arm on the far side of its endpoint and paints the same color from that endpoint through the control pixel to the grid edge.
- At crossings, yellow-control rays win; otherwise an original-vs-filled crossing uses the more fragmented pre-repair line color, while pure filled crossings use the lower color id.

Key insight: The singleton markers identify which sparse stroke fragments belong together; after they are colored, the intended lines are the axis-aligned gaps between facing endpoints, with yellow controls flipping one arm to the marked side.
</description>
</reference_puzzle>

<reference_puzzle index="11" source="eval" puzzle_id="28a6681f" lines="8-23">
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

<reference_puzzle index="12" source="eval" puzzle_id="35ab12c3" lines="73-88">
<description>
Complete outlined shapes and carry attached shadow pixels along their edges.

Input structure:
- bg = background. Each active color marks vertices of an 8-direction aligned outline.
- A start vertex can have adjacent non-bg, different-colored shadow pixels.
- Other colored pixels may be distractors unless they lie on valid drawable edges.

Rule:
- For each color, order its pixels by polar angle around their centroid to form a cycle.
- Draw only horizontal, vertical, or 45-degree edges whose interiors are clear.
- From the chosen start vertex, copy adjacent shadow offsets along each drawable edge
  when the parallel path is in bounds and clear.

Key insight: The main outline is reconstructed first, then the start vertex's neighbor offsets define parallel shadow strokes.
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

<reference_puzzle index="14" source="eval" puzzle_id="5dbc8537" lines="10-23">
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

<reference_puzzle index="15" source="eval" puzzle_id="aa4ec2a5" lines="52-66">
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

<reference_puzzle index="16" source="eval" puzzle_id="62593bfd" lines="32-46">
<description>
Sort objects to top or bottom edge by whether their shape (up to rotation/flip) is shared.

Input structure:
- bg = most common color.
- Several multicolor connected components (8-connected) scattered across the grid.

Rule:
- Compute each object's canonical shape (binary mask, invariant to rotation and flip).
- Objects whose shape appears in two or more components → translate to TOP edge (row 0).
- Objects whose shape is unique → translate to BOTTOM edge (row H−h).
- Horizontal position (left column) is preserved; only vertical translation changes.

Key insight: The sorting criterion is shape frequency (rotation-invariant), not area — duplicated shapes go up, unique shapes go down.
</description>
</reference_puzzle>

<reference_puzzle index="17" source="eval" puzzle_id="8b7bacbf" lines="42-56">
<description>
Fill enclosed regions along a line seeded by unique border pixels.

Input structure:
- bg = most common color.
- Colored lines/drawings on the grid.
- Unique single-pixel colors on the grid border mark seed endpoints of each line.

Rule:
- Each border seed pixel connects (8-connected) to a line drawing.
- Find closed bg cavities whose boundary touches that line.
- Flood-fill each such cavity with the seed pixel's color.

Key insight: Only cavities reachable through the seed's own line get filled; the seed's unique color acts as a per-line fill label.
</description>
</reference_puzzle>

<reference_puzzle index="18" source="eval" puzzle_id="21897d95" lines="283-301">
<description>
Repaint regions from T-shaped control markers, erase the markers, and optionally rotate based on a color-9 L-marker.

Input structure:
- The grid is a mosaic of monochrome regions.
- Color 1 forms T-shaped control markers on region boundaries.
- Some controls use three singleton color-1 pixels around a colored center; that center is a literal replacement color.
- An optional 5-cell L-shaped color-9 marker triggers 90-degree rotation of the output.

Rule:
- For a color-1 T marker, the single stem points at the destination region to repaint.
- The two diagonal cells behind the stem identify the source region; the destination gets that source region's original color.
- For a colored-center T marker, the same stem identifies the destination, but the center color is the replacement.
- All marker cells are filled with the final color of the source-side region.
- A 5-cell L-shaped color-9 marker controls rotation: ┐ (corner top-right) → CCW 90°, ┌ (corner top-left) → CW 90°.

Key insight:
- The stem chooses what changes, while the two diagonals behind the stem choose where the marker itself belongs.
</description>
</reference_puzzle>

<reference_puzzle index="19" source="eval" puzzle_id="247ef758" lines="35-46">
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

<reference_puzzle index="20" source="eval" puzzle_id="a47bf94d" lines="178-189">
<description>
Transmit colored tokens along wire networks, converting each to its complementary 3×3 pattern.

Input structure:
- bg = background. Non-bg cells form wire networks and 3×3 tokens (solid blocks or X/diagonal patterns) near wire endpoints.

Rule:
- Every solid 3×3 token is replaced in-place by a diamond (center + 4 cardinal neighbors).
- Each token near a wire endpoint sends its complement to the opposite endpoint: solid emits X, X emits diamond.

Key insight: Token shape cycles solid → X → diamond; wires route each complement to the far end.
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


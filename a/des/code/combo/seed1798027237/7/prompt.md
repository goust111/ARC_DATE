你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="train" puzzle_id="b745798f" lines="18-33">
<description>
Expand each 3-pixel L marker into a same-color corner L on an empty background.

Input structure
- The grid is an odd square with one dominant background color.
- Every foreground object is a monochromatic 4-connected 3-pixel L inside a
  2x2 box.

Rule
- Clear the grid to the background.
- For each marker, map its L orientation to the corresponding outer corner
  and draw that corner L in the marker color with arm length half the grid.

Key insight
- The small marker is a miniature corner; its orientation, not its position,
  chooses which outer corner receives the expanded L.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="train" puzzle_id="dc46ea44" lines="11-25">
<description>
Move the lower sprite and its pink guide into the upper panel, aligning the sprite to the guide's lower-left branch.

Input structure
- The grid has one dominant background color and one full-width separator row.
- Below the separator are exactly two foreground roles: a five-pixel guide marker and one monochrome sprite.
- The guide marker has normalized shape {(0,2), (1,2), (2,0), (2,2), (3,1)} and starts immediately below the separator.

Rule
- Clear the canvas to the background, preserving the separator row.
- Translate the guide marker upward so its top row becomes row 0.
- Move the sprite, preserving its color and shape, so the sprite bbox bottom-right aligns to the translated guide's lower-left branch.

Key insight
- The marker supplies the destination anchor: its lower-left branch cell is where the moved sprite's bbox bottom-right must land.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="eval" puzzle_id="a25697e4" lines="9-23">
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

<reference_puzzle index="4" source="train" puzzle_id="aa62e3f4" lines="18-35">
<description>
Replace the outlined object with the outside contour of that outline in the marker color.

Input structure:
- A dominant background surrounds one 8-connected foreground outline.
- The outline is the unique largest foreground color.
- One or more smaller interior colors are inside the outline; the marker color
  is the unique smallest non-outline color.

Rule:
- Erase all original foreground pixels to background.
- Paint every background pixel that is 4-adjacent to the outline with the
  marker color.
- Keep the grid shape unchanged.

Key insight: The colored interior marker supplies the paint color, while the
largest outline supplies only the contour whose outside edge is traced.
</description>
</reference_puzzle>

<reference_puzzle index="5" source="train" puzzle_id="8fff9e47" lines="8-25">
<description>
Turn a 4-cell-wide strip of 2x2 blocks into four nested corner-fill patterns.

Input structure
The input is an even-sized rectangle whose shorter dimension is four pixels.
It can be partitioned exactly into 2x2 blocks, giving one block for each layer
of the output quadrants.

Rule
Read the 2x2 blocks in reverse row-major order: bottom-right block first,
moving left, then moving upward. For each block, send its top-left,
top-right, bottom-left, and bottom-right pixels to the layer sequences of the
upper-left, upper-right, lower-left, and lower-right output quadrants.

Key insight
Each input 2x2 block supplies the four colors for one concentric layer, and
the layer order is the block order from bottom-right back to top-left.
</description>
</reference_puzzle>

<reference_puzzle index="6" source="train" puzzle_id="fd02da9e" lines="26-41">
<description>
Replace each colored corner marker with its corner-specific inward stencil in the same color.

Input structure
- The grid is 8x8 with a single dominant background color.
- Foreground pixels are marker pixels, each located at one of the four corners.
- A marker's color is carried to the stencil selected by its corner position.

Rule
- Erase each corner marker back to the background.
- Top corner markers become an inward 2x2 block one cell away from the top and side edges.
- Bottom corner markers become the mirrored three-pixel inward bent stencil shown by the bottom-corner examples.

Key insight
- The marker position selects the fixed stencil; the marker color only supplies the paint color, so multiple corners apply independently.
</description>
</reference_puzzle>

<reference_puzzle index="7" source="train" puzzle_id="252143c9" lines="14-33">
<description>
Keep only the center color and redraw it as a diagonal ray to its indicated corner.

Input structure:
- The grid is an odd square with a unique dominant background color.
- The center pixel is foreground.
- Every off-center pixel with the center color lies in the same diagonal
  quadrant relative to the center.

Rule:
- Fill the whole output with the background color.
- Use the quadrant of the center-colored pixels to choose one diagonal
  direction from the center.
- Draw a continuous line of the center color from the center to the grid
  edge in that direction.

Key insight:
The color at the exact center identifies the surviving color, while the
other pixels of that color identify which corner the center should connect to.
</description>
</reference_puzzle>

<reference_puzzle index="8" source="eval" puzzle_id="fc7cae8d" lines="9-21">
<description>
Align a central rectangle to two colored frame-edge controls via rotation/reflection.

Input structure:
- bg grid with two solid-colored border edges (distinct non-bg colors).
- A central filled rectangle with 1px colored guide edges on two sides, colors matching the frame controls.

Rule:
- Rotate/mirror the object so its guide edges face the same sides as the matching frame controls.
- Crop off the 1px guide edges from aligned sides.

Key insight: Guide-edge colors encode target frame sides; the correct dihedral transform matches both simultaneously.
</description>
</reference_puzzle>

<reference_puzzle index="9" source="train" puzzle_id="cf5fd0ad" lines="10-22">
<description>
Build a four-quadrant rotational tiling of the input square.

Input structure
- The input is a square grid treated as one complete sprite.

Rule
- Each output quadrant is a 2x2 repetition of a rotated copy of the whole input.
- The quadrant rotations are upper-left 180 degrees, upper-right 90 degrees
  clockwise, lower-left 90 degrees counterclockwise, and lower-right unchanged.

Key insight
- The output is a 4x4 grid of original-size blocks, not a per-pixel scale-up.
</description>
</reference_puzzle>

<reference_puzzle index="10" source="eval" puzzle_id="0934a4d8" lines="75-90">
<description>
Reveal the rectangular patch hidden by the solid marker using the grid's reflection symmetries.

Input structure:
- A square colored pattern with one solid rectangular marker color occluding a patch.
- The visible pattern has horizontal and vertical reflection symmetry on a large lower/right strip.
- The top and left outer strips also expose transpose counterparts for cells not recoverable by those reflections.

Rule:
- Find the unique color whose full extent is a solid rectangle; this rectangle is the output region.
- Infer the reflected strip start as the smallest row/column offset whose lower/right strips are mirror-consistent, ignoring marker pixels.
- For each marker pixel, copy the unique non-marker value in its horizontal/vertical reflection orbit; if that orbit is entirely hidden in the outer strip, use the transposed top/left-strip counterpart.

Key insight:
The marker is not part of the pattern; it covers a missing rectangle whose cells are still determined by the visible mirror or transpose copies elsewhere in the grid.
</description>
</reference_puzzle>

<reference_puzzle index="11" source="eval" puzzle_id="409aa875" lines="68-81">
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

<reference_puzzle index="12" source="train" puzzle_id="66ac4c3b" lines="57-72">
<description>
Complete the sparse-color reflected copy of the foreground motif across its matching marker row or column.

Input structure
- The grid has one background color and two foreground colors.
- One foreground color appears only as a sparse marker row or column.
- The other foreground color contains the same sparse marker positions on a parallel row or column, plus the motif on the far side of the sparse markers.

Rule
- Keep the original grid unchanged.
- Reflect every motif pixel of the larger foreground color across the two parallel marker lines.
- Paint the reflected motif pixels with the sparse marker color.

Key insight
- The two equal marker rows or columns are paired landmarks: the sparse-color marker line borders the source motif, and the matching source-color marker line anchors the reflected copy.
</description>
</reference_puzzle>

<reference_puzzle index="13" source="train" puzzle_id="d255d7a7" lines="99-114">
<description>
Move the patterned edge band to the opposite edge along every row or column whose band slice contains the trail color.

Input structure
- The dominant color is the background.
- The two foreground colors are a frequent trail color and a rarer marker color.
- All marker pixels lie nearest to one grid edge, and that edge contains trail-color runs that define the source-band thickness.

Rule
- Choose the source edge identified by the marker pixels.
- A row or column perpendicular to that edge moves if its source-band slice contains the trail color.
- Copy that slice unchanged to the opposite edge, erase the remaining trail cells to background, and keep marker-only lines fixed.

Key insight
- The marker color identifies the active edge, while the trail color identifies which parallel rows or columns carry an edge-band slice across the grid.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="train" puzzle_id="d93c6891" lines="84-101">
<description>
Move each gray bar into its adjacent orange rectangle as full stripes from the contacted edge.

Input structure
- The scene uses black outside space, yellow fill, gray one-cell-thick bars, and solid orange rectangles.
- Each gray bar 4-touches exactly one orange rectangle; some orange rectangles have no touching gray bar.
- Gray bars touching the same orange rectangle all have the same orientation.

Rule
- Recolor every original gray cell to yellow.
- For each orange rectangle touched by gray bars, count the gray cells assigned to that rectangle.
- Horizontal gray bars repaint that many cells as complete orange rows starting from the contacted top or bottom row.
- Vertical gray bars repaint that many cells as complete orange columns starting from the contacted left or right column.
- Orange rectangles with no touching gray bar remain unchanged.

Key insight
- The gray pixels are not copied by shape; their count is transferred into the neighboring orange rectangle as edge-aligned row or column stripes.
</description>
</reference_puzzle>

<reference_puzzle index="15" source="train" puzzle_id="2b9ef948" lines="51-65">
<description>
Move the yellow 3x3 frame by the displacement encoded by the gray path, then draw its diagonal yellow corner rays.

Input structure:
- The background is black.
- One 3x3 yellow frame has a non-yellow control-color center.
- A second control-color pixel is connected through gray path pixels to a lone yellow endpoint.

Rule:
- Use the vector from the second control-color pixel to the lone yellow endpoint as the frame's translation.
- Fill the whole output with the control color.
- Draw the translated yellow frame and extend yellow diagonal rays outward from its four corners to the grid edge.

Key insight: The gray path's turns are irrelevant; its two colored endpoints encode the net displacement to apply to the frame.
</description>
</reference_puzzle>

<reference_puzzle index="16" source="train" puzzle_id="bc93ec48" lines="48-63">
<description>
Copy each foreground object anchored in a true grid corner clockwise into the next corner.

Input structure
- The dominant color is the background.
- Foreground objects are monochromatic 4-connected components.
- Relevant objects are exactly those whose bounding box touches two adjacent borders at one grid corner.

Rule
- Start from the input unchanged.
- For each corner object, copy its pixel mask and color to the next corner clockwise: top-left to top-right, top-right to bottom-right, bottom-right to bottom-left, and bottom-left to top-left.
- Preserve each object's orientation by aligning its bounding box to the destination corner; stamps may overwrite existing pixels at the destination, but source pixels are not cleared.

Key insight
- Interior and side-only objects are distractors; only components actually anchored to a corner participate in the clockwise perimeter copy.
</description>
</reference_puzzle>

<reference_puzzle index="17" source="train" puzzle_id="4e7e0eb9" lines="31-49">
<description>
Transform each 9x9 panel by mirroring across its center guide line or filling all four block slots from its lone exemplar.

Input structure
The grid is made of 9x9 panels separated by one-cell-wide grid lines. Each
panel has four monochrome 3x3 slots at the quadrant positions, with either a
full non-background center row, a full non-background center column, or no
guide line.

Rule
A panel with a full center column is reflected left-to-right, and a panel
with a full center row is reflected top-to-bottom. A panel with no guide line
has exactly one slot color that appears once and one slot color that appears
three times; recolor all four slots to the once-occurring slot color.

Key insight
The center guide line encodes the mirror axis, while panels without a guide
line are completed by copying the structurally unique slot color.
</description>
</reference_puzzle>

<reference_puzzle index="18" source="train" puzzle_id="4a1cacc2" lines="8-23">
<description>
Fill the rectangle from the lone foreground pixel to its nearest grid corner.

Input structure
- The grid is a uniform background field with exactly one foreground pixel.
- The foreground pixel's color is the color used for all added pixels.

Rule
- Find the grid corner with the smallest Manhattan distance to the foreground pixel.
- Recolor every pixel in the inclusive axis-aligned rectangle bounded by that
  corner and the foreground pixel to the foreground color.

Key insight
- The single pixel is one corner of the rectangle; the opposite corner is the
  nearest corner of the whole canvas.
</description>
</reference_puzzle>

<reference_puzzle index="19" source="eval" puzzle_id="2b83f449" lines="10-22">
<description>
Simulate tokens falling through a pipe network after gates rotate open.

Input structure:
- Wall (0), bg (pipe fill), gate (horizontal 1×3 bars), token (small edge pixels).

Rule:
- Rotate each gate into a vertical blocker; gate cells become bg.
- Tokens drop downward through bg, choosing the extreme column for their side (left→min, right→max).
- Tokens settle at true pipe ends (adjacent to wall/boundary), packing inward.

Key insight: Tokens select drop columns by entry side and land at wall-bounded endpoints, ignoring blocker-bounded ones.
</description>
</reference_puzzle>

<reference_puzzle index="20" source="eval" puzzle_id="eee78d87" lines="40-55">
<description>
Decode the colored 3x3 contour motif into a 16x16 periodic lattice with the motif stamped in maroon at the center.

Input structure:
- The grid is 6x6 with orange background and one non-orange color.
- The non-orange pixels have a tight 3x3 bbox containing one contour glyph.
- The glyph is one of the task's period-3 contour forms: plus, H, X, or four corners.

Rule:
- Decode the 3x3 glyph into the period-3 orange phase tile.
- Fill a black 16x16 canvas by repeating that orange phase tile.
- Tile the original 3x3 glyph 2x2 and recolor its true pixels to maroon in the centered 6x6 region.

Key insight:
- The colored object is not copied at its input location; it is a small contour code whose phase tells where orange appears in the repeated lattice.
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
   - **不建议使用**（这些是算法量而非视觉量，会迫使解题者依赖公式而非视觉直觉）：曼哈顿 / 欧氏 / Chebyshev 距离作为精确查表键、凸包面积、二进制位编码 / 位运算、模运算 / 余数、拓扑距离（BFS / 最短路 / 跳数）、精确几何质心坐标、凸角 / 凹角计数等复杂且不易被看出来的计算。
3. 问题必须能从少量示例中推断出来（通常为 3 对训练样例 + 1 个测试样例）。设计时从人类解题者的视觉角度思考：规则涉及的所有关键要素（对象、关系、属性）必须在网格上肉眼可见，不应依赖人眼无法直接观察到的隐含信息（例如颜色id是不可用的）或需要推算的隐藏属性。规则背后的逻辑本身可以有一定挑战性，但解题所需的所有线索必须能从网格中直接获取。
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
  网格尺寸合理（≤30×30），使用颜色不超过 10 种，规则复杂度适中，人类能从 3-5 个样例中发现规律。

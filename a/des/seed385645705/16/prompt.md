你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="eval" puzzle_id="bf45cf4b" lines="7-19">
<description>
Scale up a monochrome control sprite by stamping a polychrome tile at each filled pixel.

Input structure:
- bg = background color.
- Two foreground objects (8-connected): a monochrome rectangular sprite with bg holes (control), and a polychrome square tile (block).

Rule:
- For each cell in the control sprite's bbox: if filled, stamp a copy of the block tile; if bg, leave a tile-sized bg region.
- Output grid dimensions are (control_h × tile_h) by (control_w × tile_w).

Key insight: The control sprite acts as a binary mask that gets "pixel-scaled" by the tile — each pixel expands to a full tile copy or empty space.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="train" puzzle_id="1d61978c" lines="8-22">
<description>
Recolor the two diagonal foreground-bar orientations, assigning teal to the larger orientation and red to the smaller one.

Input structure
- The grid has one dominant background color and one foreground color.
- Foreground pixels form one-cell-thick diagonal bars.
- Every foreground pixel belongs to exactly one diagonal orientation.

Rule
- Keep the background unchanged.
- Classify each foreground pixel by whether it has a diagonal neighbor on the NW/SE axis or the NE/SW axis.
- Recolor the orientation with more foreground pixels teal and the other orientation red.

Key insight
- The output colors depend on the relative population of the two diagonal orientations, not on a fixed slash direction or position.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="train" puzzle_id="833966f4" lines="8-22">
<description>
Swap the upper two-pixel segment and the lower two-pixel segment around the fixed center pixel.

Input structure:
- The grid is a 5x1 vertical column of individual colored pixels.
- The middle pixel is a separator between two same-size two-pixel segments.

Rule:
- Keep the middle pixel in place.
- Reverse the order of the two pixels above the middle.
- Reverse the order of the two pixels below the middle.

Key insight:
- The transformation is local vertical order reversal on each side of the center, not a color-specific rule.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="train" puzzle_id="e729b7be" lines="8-20">
<description>
Complete the scene by adding the point-reflected copy of the colored object across the central marker.

Input structure
- The grid has odd dimensions, a dominant background, and a center-row marker color appearing only at the left edge, center, and right edge.
- A single divider color fills the center column except for the center marker.
- All remaining foreground pixels form the object to copy, entirely on the left side of the divider.
Rule
- Keep the input unchanged.
- For every object pixel, paint the pixel rotated 180 degrees around the center marker with the same color.
Key insight
- The center marker is the rotation center; the yellow-looking divider is not mirrored, it only separates the source from the empty target side.
</description>
</reference_puzzle>

<reference_puzzle index="5" source="train" puzzle_id="97c75046" lines="88-107">
<description>
Move the single marker to the endpoint reached by sliding along the obstacle contour.

Input structure:
- A dominant background color fills the canvas.
- One single-pixel marker sits away from a connected obstacle region of the
  remaining color.
- The marker has exactly one horizontal or vertical sightline to the obstacle.

Rule:
- Find the last background pixel before the obstacle along that sightline.
- From that pixel, try the two diagonal turns that continue around the
  obstacle boundary.
- Move the marker to the endpoint of the turn that remains 4-adjacent to the
  obstacle for the longest distance.

Key insight:
The marker is not placed by distance to the obstacle; it first points to the
obstacle, then follows the visible outside contour until that contour segment ends.
</description>
</reference_puzzle>

<reference_puzzle index="6" source="eval" puzzle_id="7b5033c1" lines="7-21">
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

<reference_puzzle index="7" source="train" puzzle_id="fe45cba4" lines="72-89">
<description>
Move the left-edge fragment into its same-colored anchor fragment to complete a solid rectangle.

Input structure
- The grid has a dominant background color.
- Exactly one foreground color is split into two 4-connected objects: one
  touches the left edge, and the other is an anchor away from the left edge.
- The translated left-edge object and the anchor can form one filled rectangle.
Rule
- Find the unique split color whose left-edge object can be translated onto
  background cells beside the other same-colored object so their union is a
  solid axis-aligned rectangle.
- Clear the left-edge object's original cells to the background.
- Paint that object at the unique completing offset, preserving all other cells.
Key insight
- The active color is identified by object structure and rectangle completion,
  not by a fixed color id; the same-colored anchor object stays in place.
</description>
</reference_puzzle>

<reference_puzzle index="8" source="train" puzzle_id="337b420f" lines="41-55">
<description>
Keep the larger sprite from each separated panel and overlay those sprites into one 5x5 canvas.

Input structure:
- The input is a 5x17 grid: three 5x5 panels separated by one-cell black columns.
- Each panel has the same background color and exactly one foreground color split into two 4-connected components.

Rule:
- In each panel, discard the smaller foreground component and keep the unique largest component as the sprite.
- Process panels left to right, placing each sprite at its panel coordinates on a 5x5 background canvas.
- If a sprite would cover an already placed sprite, slide it by the unique nearest in-bounds offset that avoids overlap.

Key insight:
- The extra same-colored component in each panel is a distractor; the final answer is the non-overlapping overlay of the three largest components.
</description>
</reference_puzzle>

<reference_puzzle index="9" source="train" puzzle_id="95755ff2" lines="45-60">
<description>
Paint straight rays from colored side markers into the hollow diamond frame until an interior foreground pixel blocks each ray.

Input structure
- The background contains one complete one-pixel-thick diamond frame.
- Colored marker pixels outside the diamond sit on the top, bottom, left, or right side of the frame's bounding box.
- Any non-background, non-frame pixels inside the diamond are blockers that remain unchanged.

Rule
- Each outside marker emits inward perpendicular to its side of the diamond bounding box.
- Along that row or column, paint only background pixels strictly inside the diamond with the marker color.
- Stop a ray when it reaches an existing interior foreground pixel; keep the frame and all original colored pixels unchanged.

Key insight
- The diamond frame defines a strict interior, while the side markers are column or row emitters whose color fills the visible interior interval until blocked.
</description>
</reference_puzzle>

<reference_puzzle index="10" source="train" puzzle_id="37ce87bb" lines="33-47">
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

<reference_puzzle index="11" source="eval" puzzle_id="5961cc34" lines="8-21">
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

<reference_puzzle index="12" source="eval" puzzle_id="7c66cb00" lines="9-24">
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

<reference_puzzle index="13" source="eval" puzzle_id="d8e07eb2" lines="29-47">
<description>
Look up query glyphs in a reference table and mark whether they form a line.

Input structure:
- Teal background with two full-width pink separator rows.
- Above the first separator is a query row of 5x5 glyph cells.
- Between separators is a reference table of the same 5x5 cells; a glyph is
  identified by both its color and its 3x3 shape.

Rule:
- Green-fill the background inside every table cell whose glyph exactly
  appears in the query row.
- If the matched table cells form one complete row or one complete column,
  also green-fill the query band and the bottom verdict band.
- Otherwise leave the query band unchanged and fill the bottom verdict band red.

Key insight: The bottom color is not a count; it is the yes/no answer to
whether the queried glyphs align as a full row or column in the table.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="eval" puzzle_id="a395ee82" lines="9-24">
<description>
Stamp translated copies of a monochrome template guided by a singleton pixel layout.

Input structure:
- bg = background. A large monochrome shape (template, color T).
- Singleton pixels elsewhere: exactly one is color T (layout origin); the rest share color O (overlap color).
- Singletons sit on a uniform-spacing lattice.

Rule:
- Convert each singleton's offset from the origin into coarse lattice units.
- Place a template copy at each offset, scaled by template height/width.
- The origin's copy is painted O instead of T.
- Any pixel covered by two or more copies is also recolored to O.

Key insight: The single T-colored singleton identifies both the origin position and the "self" copy, which is uniquely recolored.
</description>
</reference_puzzle>

<reference_puzzle index="15" source="eval" puzzle_id="65b59efc" lines="9-24">
<description>
Decode a stamp map: read templates, markers, and palette from a tiled grid, then render stamps.

Input structure:
- Grid divided into K tile columns (each N wide) separated by 1-col dashed separators.
- Top band (rows 0..N-1): K template masks, one per tile column.
- Middle band (rows N+1..2N): marker pixels whose color picks a template.
- Bottom band (rows 2N+2..): K palette swatches giving the output color per template.
- Separator rows at rows N and 2N+1 (dashed pattern of sep color).

Rule:
- Each marker's local position (lr, lc) within its tile column determines which N×N output block to stamp.
- The marker's color identifies which template to use; the palette recolors the template.

Key insight: Marker color encodes template index; local position within tile column encodes stamp block.
</description>
</reference_puzzle>

<reference_puzzle index="16" source="train" puzzle_id="f8cc533f" lines="10-25">
<description>
Complete every colored sprite by applying the union of all aligned partial sprite masks.

Input structure:
- The dominant color is the background.
- Each non-background color appears as a separated partial copy of one common sprite mask.
- The copies share a top-left-aligned sprite window; some copies may be missing lower or right edge pixels.

Rule:
- Crop each same-color mask to its bounding box.
- Pad the crops to the maximum observed height and width, aligned at the top-left, and OR them into one complete template.
- Fill that template at every sprite location using the sprite's own color.

Key insight:
- No single input sprite is complete, but their missing cells are complementary, so the aligned union reveals the full mask.
</description>
</reference_puzzle>

<reference_puzzle index="17" source="eval" puzzle_id="b0039139" lines="29-44">
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

<reference_puzzle index="18" source="train" puzzle_id="db7260a4" lines="37-56">
<description>
Drop the singleton marker into the matching red U-shaped container, or make a bottom-row puddle if no closed container is hit.

Input structure
- The grid has a dominant background, one singleton marker pixel on the top row,
  and one other foreground color forming vertical bars and bottom segments.
- A closed container is a U shape: two vertical side bars with a continuous
  bottom segment between them and an empty rectangular interior above the bottom.

Rule
- Remove the top marker from its original cell.
- If exactly one closed U-shaped container has the marker column within its
  horizontal footprint, fill that container's empty interior with the marker color.
- If no closed container is under the marker column, fill the whole bottom row
  with the marker color.

Key insight
- The marker does not identify a color by value; it identifies a vertical drop
  column, and only a structurally closed U shape can catch and contain the fill.
</description>
</reference_puzzle>

<reference_puzzle index="19" source="train" puzzle_id="99caaf76" lines="56-75">
<description>
Move each marked cross object away from its marker to the grid edge while rotating the colored cross 180 degrees.

Input structure:
- The background is the dominant color.
- Each 8-connected foreground object has nine pixels: a 4-pixel marker-color
  sprite attached to a 5-pixel non-marker colored cross.
- Each object bbox is either 3x6 or 6x3, so the marker sprite is on exactly
  one side of the cross.

Rule:
- Move the whole object straight away from its marker side until the object
  bbox touches the corresponding grid edge.
- Keep the marker sprite unchanged within the object.
- Rotate the 3x3 colored cross block by 180 degrees before placing it.

Key insight:
The marker sprite is a directional tail; the cross is the leading payload,
and only that payload turns around during the slide to the edge.
</description>
</reference_puzzle>

<reference_puzzle index="20" source="eval" puzzle_id="8698868d" lines="9-23">
<description>
Match sprites to frames by connected-hole count, then nest each sprite inside its frame.

Input structure:
- bg = azure (8). Large hollow squares (frames) arranged in a grid; smaller filled squares (sprites) placed elsewhere.
- Each frame and sprite has a distinct color. Sprites may contain holes (non-sprite-color pixels).

Rule:
- Frames form a regular grid; some may be bg-colored (invisible).
- Count connected hole regions inside each frame's interior and each sprite.
- Pair each frame with the sprite sharing the same hole count.
- Place the matched sprite inside the frame border; recolor sprite holes to the frame's color.

Key insight: The matching key is the number of connected hole components, not hole area or color.
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


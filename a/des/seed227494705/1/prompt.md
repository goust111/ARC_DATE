你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="train" puzzle_id="54dc2872" lines="52-69">
<description>
Move each marked L-shaped sprite onto the separate pixel matching its marker color.

Input structure
- The background is the dominant color.
- Foreground has 4-connected multicolor sprites and isolated one-pixel controls.
- Each sprite has one repeated body color in an L shape plus one embedded marker
  pixel of a different color.

Rule
- If the embedded marker color also appears as a separate isolated control pixel,
  translate the whole sprite so the body L-bend lands on that control pixel.
- Clear each moved sprite from its old position, clear its consumed control pixel,
  then paste all translated sprites; sprites without a matching control stay put.

Key insight
- The marker color pairs a sprite with its destination control, while the L-bend
  gives the orientation-independent anchor cell to place on that destination.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="train" puzzle_id="4a1cacc2" lines="8-23">
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

<reference_puzzle index="3" source="train" puzzle_id="9841fdad" lines="15-39">
<description>
Expand the colored runs from the left reference panel into the blank right panel.

Input structure:
- A uniform rectangular frame surrounds the grid.
- One full-height divider of the frame color splits a left reference panel
  from a blank right panel.
- Both panels have one background margin column on each side of their active
  columns; the left active columns contain two-column endpoint slots, possibly
  connected by same-color horizontal runs.

Rule:
- Keep the original grid and paint only in the right panel's active columns.
- A source run exactly matching the left endpoint slot is copied to the left
  two columns of the right active span.
- A source run exactly matching the right endpoint slot is copied to the right
  two columns of the right active span.
- A source run connecting the endpoint slots is stretched across the entire
  right active span.

Key insight:
The left panel is a compressed legend for the right panel: endpoint markers
stay two pixels wide, while connectors between endpoints expand to fill the
distance between them.
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

<reference_puzzle index="5" source="eval" puzzle_id="21897d95" lines="283-301">
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

<reference_puzzle index="6" source="train" puzzle_id="7d7772cc" lines="70-92">
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

<reference_puzzle index="7" source="eval" puzzle_id="800d221b" lines="7-22">
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

<reference_puzzle index="8" source="eval" puzzle_id="80a900e0" lines="69-82">
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

<reference_puzzle index="9" source="train" puzzle_id="a834deea" lines="20-39">
<description>
Fill each black 5x5 region's missing interior positions with a fixed 3x3 positional color legend.

Input structure
- The grid uses teal as the dominant background and black for one or more
  5x5 regions.
- Each black region has a black one-pixel border and a 3x3 interior that
  may already contain teal pixels.

Rule
- Overlay the legend 1 7 6 / 4 8 5 / 2 9 3 on the 3x3 interior of every
  5x5 black region.
- Wherever an interior non-center legend position is still black, recolor it
  to the corresponding legend color; leave existing teal pixels, the center,
  and the black border unchanged.

Key insight
- The colored output pixels are not derived from the teal motif shape; they
  are the fixed position labels for the missing cells of each 3x3 interior.
</description>
</reference_puzzle>

<reference_puzzle index="10" source="eval" puzzle_id="8698868d" lines="9-23">
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

<reference_puzzle index="11" source="train" puzzle_id="25e02866" lines="14-31">
<description>
Merge same-sized filled rectangular objects by overlaying their internal marker pixels.

Input structure:
- A dominant background surrounds two or more separated 4-connected foreground objects.
- Every object is a filled rectangle of the same size.
- Each rectangle has the same majority base color plus one or more non-base marker pixels.

Rule:
- Create one output rectangle with the shared object size, filled with the shared base color.
- For each input rectangle, copy every non-base marker pixel to the same relative location
  in the output.
- Marker overlays must either land on base color or agree in color with an existing marker.

Key insight:
The separated rectangles are layers of the same template; their colored marker pixels
combine by local coordinates while the repeated base fill is just the output canvas.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="train" puzzle_id="ad173014" lines="97-111">
<description>
Recolor each framed interior mask with the color from the next red frame clockwise around the blue connector cycle.

Input structure
- The grid contains hollow red square frames connected pairwise by blue line components.
- The blue components connect the frames into one cycle, with every frame touching exactly two blue components.
- Each frame interior contains background plus exactly one non-background color mask.

Rule
- Sort the red frames clockwise by their centers around the cycle.
- Keep each frame's interior mask shape in place, but recolor it with the interior color from the next clockwise frame.
- Leave the red frames, blue connectors, background, and interior background holes unchanged.

Key insight
- The colored masks do not move; the blue cycle gives the order in which their colors rotate.
</description>
</reference_puzzle>

<reference_puzzle index="13" source="train" puzzle_id="c9680e90" lines="52-67">
<description>
Move each mirrored bottom source pixel to the far end of its trail and redraw the top half as its reflection.

Input structure
- A solid non-background row separates equal-height top and bottom halves.
- The top half has one marker color on the background.
- In the bottom half, exactly one color has positions that mirror the top markers across the separator, and the other color forms trails touching those source pixels.

Rule
- Identify the bottom source color by reflecting each bottom color into the top half and matching the top marker set.
- For every bottom source pixel, follow its unique 4-connected trail of the other bottom color to the trail endpoint.
- Clear the old top markers, bottom sources, and trails; place the source color at each endpoint and place the top marker color at the reflected endpoint.

Key insight
- The top half is only a mirror display of the bottom moving sources; the actual motion is encoded by the trail attached to each bottom source pixel.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="train" puzzle_id="84ba50d3" lines="36-54">
<description>
Drop the blue sprites downward relative to the red divider, with vertical sprites passing through to the bottom.

Input structure:
- The background is the dominant color.
- One non-background color forms a single full-width horizontal divider row.
- The other non-background color forms 4-connected sprites above the divider.

Rule:
- Clear the original sprite pixels.
- A one-column sprite falls to the bottom of its column and deletes the divider
  pixel it crossed.
- Any wider sprite falls until its unique widest horizontal row is immediately
  above the divider row; if the moved sprite overlaps the divider, it covers it.

Key insight:
The red row is not a generic floor: only one-column sprites pass through to the
bottom, while wider sprites are registered by their widest horizontal segment.
</description>
</reference_puzzle>

<reference_puzzle index="15" source="eval" puzzle_id="d59b0160" lines="39-54">
<description>
Delete connected components whose colors match a legend swatch.

Input structure:
- Background (bg) fills most of the grid.
- Top-left corner: a small square legend region bordered on its bottom and right edges by a single color; the interior holds a set of "target" colors.
- Elsewhere: multicolored objects (4-connected components) on the bg.

Rule:
- Read the target color set from inside the legend border.
- For each connected component, check whether it contains **all** target colors.
- Erase (fill with bg) every component that does; leave all others unchanged.
- Output grid is the same size.

Key insight: The legend acts as a filter specification — only components carrying the full color palette shown in the legend are removed.
</description>
</reference_puzzle>

<reference_puzzle index="16" source="train" puzzle_id="7e2bad24" lines="91-106">
<description>
Extend the diagonal blue ray through the grid, reflecting it off red line-segment mirrors.

Input structure
- Black-background grid with one diagonal 8-connected blue ray seed.
- Red foreground objects, when present, are straight horizontal or vertical mirror segments.
- Other foreground bars are transparent to the ray and may be overwritten where the ray crosses them.

Rule
- Start from the ray seed endpoint whose outward diagonal extension remains inside the grid.
- Paint the ray continuously in that diagonal direction until it leaves the grid.
- If the next cell is a red mirror, do not paint it; reflect by flipping the row direction on a horizontal mirror or the column direction on a vertical mirror.

Key insight
- The non-red bars are decoys: only red bars reflect the diagonal path, while all other crossed cells become blue.
</description>
</reference_puzzle>

<reference_puzzle index="17" source="train" puzzle_id="e45ef808" lines="31-46">
<description>
Mark the shallowest and deepest columns of a solid terrain boundary with vertical colored guide lines.

Input structure:
- The first row is a solid empty separator.
- Below it, each column consists of upper fill cells followed by a contiguous suffix of terrain cells.
- The bottom row is all terrain, so every column has a well-defined first terrain row.

Rule:
- Find the column whose first terrain row is latest; it is unique and its upper fill run becomes maroon.
- Find the rightmost column whose first terrain row is earliest; its upper fill run becomes yellow.
- Preserve the empty separator, terrain cells, and all other upper fill cells.

Key insight:
- The task is a column-height summary of the terrain profile: color the deepest upper run and the highest terrain peak.
</description>
</reference_puzzle>

<reference_puzzle index="18" source="train" puzzle_id="f28a3cbb" lines="42-50">
<description>
Compress each colored ray pixel inward until it touches that color's 3x3 anchor block.
Input structure
- A uniform background grid with non-background colors.
- Each foreground color contains one solid 3x3 anchor block; all other pixels of that color lie on straight 8-neighbor rays aimed at that block.
Rule
- Keep anchor blocks fixed, erase loose foreground pixels, and for each ray pack its pixels into consecutive cells adjacent to the anchor along the same ray.
Key insight
- The scattered pixels are displaced pixels on visible row, column, or diagonal sightlines back to their same-colored 3x3 block.
</description>
</reference_puzzle>

<reference_puzzle index="19" source="train" puzzle_id="4df5b0ae" lines="27-41">
<description>
Pack the colored sprites along the bottom edge from smallest to largest.

Input structure
The grid has a dominant background color and several monochrome 4-connected
foreground sprites; a full-grid one-cell frame may surround the scene.

Rule
Remove any full-grid rectangular frame, sort the remaining sprites by pixel
area from smallest to largest, and paste their bounding-box sprites left to
right with each sprite bottom-aligned.

Key insight
The original locations do not determine the final slots; each sprite's area
gives the packing order, while its shape and color are preserved.
</description>
</reference_puzzle>

<reference_puzzle index="20" source="eval" puzzle_id="7b0280bc" lines="54-69">
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


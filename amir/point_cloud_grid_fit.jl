# using CairoMakie, DelimitedFiles

# data = readdlm("amir/vel_2Daneu_crop.csv", ',', Float64)

# data = data[1:3200, :]

# v_x, v_y = data[1:2:end, :], data[2:2:end, :]

# v = [collect.(zip(vt[1], vt[2])) for vt in zip(v_x, v_y)]

# v0 = reshape(v[:, 1], (40, :))

# xs = range(-5, 5, 40)
# ys = range(-5, 5, 40)
# us = first.(v0)
# vs = last.(v0)
# strength = sqrt.(us .^ 2 .+ vs .^ 2)

# arrows(xs, ys, us, vs, lengthscale=0.005, arrowcolor=strength)

using CairoMakie, CSV, DataFrames

v0_data = CSV.File("../../cropped_2D_aneurysm/vel_2Daneu_crop.0.csv") |> DataFrame

xs, ys = v0_data."Points:0", v0_data."Points:1"
us, vs = v0_data."f_27:0", v0_data."f_27:1"
norms(v) = sqrt(sum(v.^2))
all_uv_norms = norms.(collect(zip(us, vs)))

arrows(xs, ys, us, vs, lengthscale=0.005, arrowcolor=all_uv_norms, linecolor=all_uv_norms)

point_range = 1500
xs_grid = range(minimum(xs), maximum(xs), length=point_range)
ys_grid = range(minimum(ys), maximum(ys), length=point_range)
grid_points = collect(Iterators.product(xs_grid, ys_grid))
points = collect(zip(xs, ys, us, vs))
points_unqiue = unique(points)
dist(p1, p2) = sqrt(sum((p1 .- p2).^2))
grid_p_idx_to_p = Dict()
for (i, p) in enumerate(points_unqiue)
    distances = (x->dist((p[1], p[2]), x)).(grid_points)
    closest_p_idx = argmin(distances)
    if !haskey(grid_p_idx_to_p, closest_p_idx)
        grid_p_idx_to_p[closest_p_idx] = []
    end
    push!(grid_p_idx_to_p[closest_p_idx], p)
end

# scatter!(Point2f.(vec(grid_points)))
# current_figure()

grid_velocities = Matrix{Tuple{Float64, Float64}}(undef, point_range, point_range)
for (i, p) in grid_p_idx_to_p
    grid_velocities[i] = (last(p)[3], last(p)[4])
end

uv_norms = norms.(grid_velocities)
heatmap(uv_norms, colormap=:heat)
# precision of Point2f too low? just dont use it when exporting final matrix
filter_zero_norm(v) = [Point2f(p[1]) for p in zip(vec(v), vec(uv_norms)) if p[2] != 0]
grid_points_to_plot = filter_zero_norm(grid_points)
grid_velocities_to_plot = filter_zero_norm(grid_velocities)
uv_norms_to_plot = filter(x->x != 0, uv_norms)
arrows(grid_points_to_plot, grid_velocities_to_plot, lengthscale=0.005, arrowcolor=uv_norms_to_plot, linecolor=uv_norms_to_plot)

for (loc, v) in zip(vec.([grid_points, grid_velocities])...)
# for i in range(1, length(grid_points))
    # loc, v = grid_points[i], grid_matrix[i]
    if norms(v) != 0
        # @show v, norms(v)
        arrows!([Point2f(loc)], [Point2f(v)], lengthscale=0.01)#, arrowcolor=[norms(v)], linecolor=[norms(v)])
    end
end
current_figure()

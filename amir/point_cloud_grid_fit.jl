using CairoMakie, CSV, DataFrames, Statistics

function l2norm(v)
    sqrt(sum(v.^2))
end

function grid_point_bucketing(lattice, data_points)
    lattice_point_to_data_point = Dict()
    for dp in data_points
        dp_coordinate = dp[1], dp[2]
        distances = (lp->l2norm.(dp_coordinate .- lp)).(lattice)
        closest_lattice_point_index = argmin(distances)
        push!(get!(lattice_point_to_data_point, closest_lattice_point_index, []), dp)
    end
    lattice_point_to_data_point
end

function get_neighbor_indices(i, matrix)
    x, y = Tuple(i)
    indices = [
        (x, y - 1), (x, y + 1),
        (x + 1, y), (x + 1, y + 1), (x + 1, y - 1),
        (x - 1, y), (x - 1, y + 1), (x - 1, y - 1)
    ]
    matrix_size = size(matrix)
    in_bounds(i) = 1 ≤ i[1] ≤ matrix_size[1] && 1 ≤ i[2] ≤ matrix_size[2]
    cart_indices = CartesianIndex.(filter(i->in_bounds(i), indices))
    cart_indices
end

function set_neighbors!(i, v, matrix)
    cart_indices = get_neighbor_indices(i, matrix)
    cart_indices = filter(i->l2norm(matrix[i]) < 1e-10, cart_indices)
    for ci in cart_indices
        matrix[ci] = v
    end
end

function copy_avg_neighbors!(i, matrix)
    cart_indices = get_neighbor_indices(i, matrix)
    cart_indices = filter(i->l2norm(matrix[i]) > 1e-10, cart_indices)
    if l2norm(matrix[i]) < 1e-10 && length(cart_indices) > 0
        matrix_values = map(i->matrix[i], cart_indices)
        avg_vector = (mean(first.(matrix_values)), mean(last.(matrix_values)))
        matrix[i] = avg_vector
    end
end

function copy_vectors!(matrix)
    for (i, v) in pairs(matrix)
        # set_neighbors!(i, v, matrix)
        copy_avg_neighbors!(i, matrix)
    end
end

v0_data = CSV.File("amir/vel_2Daneu_crop.0.csv") |> DataFrame

xs, ys = v0_data."Points:0", v0_data."Points:1"
us, vs = v0_data."f_27:0", v0_data."f_27:1"
all_uv_norms = l2norm.(collect(zip(us, vs)))

# arrows(xs, ys, us, vs, lengthscale=0.005, arrowcolor=all_uv_norms, linecolor=all_uv_norms)

num_points = 500
x_lattice = range(minimum(xs), maximum(xs), length=num_points)
y_lattice = range(minimum(ys), maximum(ys), length=num_points)
lattice = collect(Iterators.product(x_lattice, y_lattice))
data_points = unique(collect(zip(xs, ys, us, vs)))

lattice_point_to_data_point = grid_point_bucketing(lattice, data_points)

lattice_velocities = Matrix{Tuple{Float64, Float64}}(undef, num_points, num_points)
for (i, p) in lattice_point_to_data_point
    lattice_velocities[i] = (last(p)[3], last(p)[4])
end

copy_vectors!(lattice_velocities)

uv_norms = l2norm.(lattice_velocities)
heatmap(uv_norms)#, colormap=:heat)
filter_zero_norm(v) = [Point2f(p[1]) for p in zip(vec(v), vec(uv_norms)) if p[2] > 1e-10]
lattice_points_to_plot = filter_zero_norm(lattice)
lattice_velocities_to_plot = filter_zero_norm(lattice_velocities)
uv_norms_to_plot = filter(x->x > 1e-10, uv_norms)
arrows(lattice_points_to_plot, lattice_velocities_to_plot, lengthscale=0.005, arrowcolor=uv_norms_to_plot, linecolor=uv_norms_to_plot)

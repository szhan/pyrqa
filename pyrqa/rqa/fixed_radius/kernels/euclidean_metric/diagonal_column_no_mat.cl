/*
This file is part of PyRQA.
Copyright 2015 Tobias Rawald, Mike Sips.
*/

__kernel void diagonal_symmetric(
    __global const float* time_series_x,
    __global const float* time_series_y,
    const uint dim_x,
    const uint dim_y,
    const uint start_x,
    const uint start_y,
    const uint m,
    const uint t,
    const float e,
    const uint w,
    const uint offset,
    __global uint* frequency_distribution,
    __global uint* carryover
)
{
    uint global_id_x = get_global_id(0);

    uint id_x = global_id_x + offset;
    uint id_y = 0;

    if (id_x < dim_x && abs_diff(start_x + id_x, start_y + id_y) >= w)
    {
        uint t_x;
        uint t_y;
        float sum;

        uint carryover_id = id_x;
        if (offset > 0)
        {
            carryover_id = dim_x - id_x;
        }

        uint buffer = carryover[carryover_id];

        while (id_x < dim_x && id_y < dim_y)
        {
            sum = 0.0f;
            for (uint i = 0; i < m; ++i)
            {
                t_x = id_x + (i * t);
                t_y = id_y + (i * t);

                sum += (time_series_x[t_x] - time_series_y[t_y]) * (time_series_x[t_x] - time_series_y[t_y]);
            }

            if (sum < e*e)
            {
                buffer++;
            }
            else
            {
                if(buffer > 0)
                {
                    atomic_inc(&frequency_distribution[buffer - 1]);
                }

                buffer = 0;
            }

            id_x++;
            id_y++;
        }

        carryover[carryover_id] = buffer;
    }
}

/*
This file is part of PyRQA.
Copyright 2015 Tobias Rawald, Mike Sips.
*/

__kernel void clear_buffer(
    __global uint* buffer
)
{
    buffer[get_global_id(0)] = 0;
}

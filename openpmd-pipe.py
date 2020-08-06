#!/usr/bin/env python3

import openpmd_api as io
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='openPMD Pipe')

    parser.add_argument('--infile', type=str, help='In file')
    parser.add_argument('--outfile', type=str, help='Out file')
    parser.add_argument('--inconfig',
                        type=str,
                        default='{}',
                        help='JSON config for the in file')
    parser.add_argument('--outconfig',
                        type=str,
                        default='{}',
                        help='JSON config for the out file')

    return parser.parse_args()


class pipe:
    def __init__(self, infile, outfile, inconfig, outconfig):
        self.infile = infile
        self.outfile = outfile
        self.inconfig = inconfig
        self.outconfig = outconfig
        self.chunks = []

    def run(self):
        inseries = io.Series(self.infile, io.Access_Type.read_only,
                             self.inconfig)
        outseries = io.Series(self.outfile, io.Access_Type.create,
                              self.outconfig)
        write_iterations = outseries.write_iterations()
        print("rabimmel", type(inseries), isinstance(inseries, io.Series))
        for in_iteration in inseries.read_iterations():
            print("Iteration {0} contains {1} meshes:".format(
                in_iteration.iteration_index, len(in_iteration.meshes)))
            for m in in_iteration.meshes:
                print("\t {0}".format(m))
            print("")
            print("Iteration {0} contains {1} particle species:".format(
                in_iteration.iteration_index, len(in_iteration.particles)))
            for ps in in_iteration.particles:
                print("\t {0}".format(ps))
                print("With records:")
                for r in in_iteration.particles[ps]:
                    print("\t {0}".format(r))
            out_iteration = write_iterations[in_iteration.iteration_index]
            self.__copy(in_iteration, out_iteration)
            in_iteration.close()
            out_iteration.close()
            self.chunks.clear()

    def __copy(self, src, dest):
        if (type(src) != type(dest)
                and not isinstance(src, io.IndexedIteration)
                and not isinstance(dest, io.Iteration)):
            raise RuntimeError(
                "Internal error: Trying to copy mismatching types")
        container_types = [
            io.Mesh_Container, io.Particle_Container, io.ParticleSpecies,
            io.Record, io.Mesh
        ]
        if isinstance(src, io.Iteration):
            self.__copy(src.meshes, dest.meshes)
            self.__copy(src.particles, dest.particles)
        elif any([
                isinstance(src, container_type)
                for container_type in container_types
        ]):
            for key in src:
                self.__copy(src[key], dest[key])
        elif isinstance(src, io.Record_Component):
            shape = src.shape
            dtype = src.dtype
            dest.reset_dataset(io.Dataset(dtype, shape))
            chunk = src.load_chunk([0 for _ in shape], shape)
            self.chunks.append(chunk)
            dest[()] = chunk


if __name__ == "__main__":
    args = parse_args()
    pipe = pipe(args.infile, args.outfile, args.inconfig, args.outconfig)
    pipe.run()

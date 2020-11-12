#!/usr/bin/env python3

from mpi4py import MPI
import openpmd_api as io
import argparse
import sys  # sys.stderr.write

debug = False


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


class Chunk:
    """
    A Chunk is an n-dimensional hypercube, defined by an offset and an extent.
    Offset and extent must be of the same dimensionality (Chunk.__len__).
    """
    def __init__(self, offset, extent):
        assert (len(offset) == len(extent))
        self.offset = offset
        self.extent = extent

    def __len__(self):
        return len(self.offset)

    def slice1D(self, mpi_rank, mpi_size, dimension=None):
        """
        Slice this chunk into mpi_size hypercubes along one of its n dimensions.
        The dimension is given through the 'dimension' parameter. If None, the
        dimension with the largest extent on this hypercube is automatically
        picked.
        Returns the mpi_rank'th of the sliced chunks.
        """
        if dimension is None:
            # pick that dimension which has the highest count of items
            dimension = 0
            maximum = self.extent[0]
            for k, v in enumerate(self.extent):
                if v > maximum:
                    dimension = k
        assert (dimension < len(self))
        # no offset
        assert (self.offset == [0 for _ in range(len(self))])
        offset = [0 for _ in range(len(self))]
        stride = self.extent[dimension] // mpi_size
        rest = self.extent[dimension] % mpi_size

        # local function f computes the offset of a rank
        # for more equal balancing, we want the start index
        # at the upper gaussian bracket of (N/n*rank)
        # where N the size of the dataset in dimension dim
        # and n the MPI size
        # for avoiding integer overflow, this is the same as:
        # (N div n)*rank + round((N%n)/n*rank)
        def f(rank):
            res = stride * rank
            padDivident = rest * rank
            pad = padDivident // mpi_size
            if pad * mpi_size < padDivident:
                pad += 1
            return res + pad

        offset[dimension] = f(mpi_rank)
        extent = self.extent.copy()
        if mpi_rank >= mpi_size - 1:
            extent[dimension] -= offset[dimension]
        else:
            extent[dimension] = f(mpi_rank + 1) - offset[dimension]
        return Chunk(offset, extent)


class pipe:
    """
    Represents the configuration of one "pipe" pass.
    """
    def __init__(self, infile, outfile, inconfig, outconfig, comm):
        self.infile = infile
        self.outfile = outfile
        self.inconfig = inconfig
        self.outconfig = outconfig
        self.chunks = []
        self.comm = comm

    def run(self):
        if self.comm.size == 1:
            print("Opening data source")
            sys.stdout.flush()
            inseries = io.Series(self.infile, io.Access_Type.read_only,
                                 self.inconfig)
            print("Opening data sink")
            sys.stdout.flush()
            outseries = io.Series(self.outfile, io.Access_Type.create,
                                  self.outconfig)
            print("Opened input and output")
            sys.stdout.flush()
        else:
            print("Opening data source")
            sys.stdout.flush()
            inseries = io.Series(self.infile, io.Access_Type.read_only,
                                 self.comm, self.inconfig)
            print("Opening data sink")
            sys.stdout.flush()
            outseries = io.Series(self.outfile, io.Access_Type.create,
                                  self.comm, self.outconfig)
            print("Opened input and output")
            sys.stdout.flush()
        self.__copy(inseries, outseries)

    def __copy(self, src, dest, current_path="/data/"):
        """
        Worker method.
        Copies data from src to dest. May represent any point in the openPMD
        hierarchy, but src and dest must both represent the same layer.
        """
        if (type(src) != type(dest)
                and not isinstance(src, io.IndexedIteration)
                and not isinstance(dest, io.Iteration)):
            raise RuntimeError(
                "Internal error: Trying to copy mismatching types")
        for key in src.attributes:
            if key == "openPMDextension":
                # this sets the wrong datatype otherwise
                dest.set_openPMD_extension(src.openPMD_extension)
            else:
                attr = src.get_attribute(key)
                if key == "unitDimension":
                    if hasattr(dest, 'unit_dimension'):
                        dest.unit_dimension = {
                            io.Unit_Dimension.L: attr[0],
                            io.Unit_Dimension.M: attr[1],
                            io.Unit_Dimension.T: attr[2],
                            io.Unit_Dimension.I: attr[3],
                            io.Unit_Dimension.theta: attr[4],
                            io.Unit_Dimension.N: attr[5],
                            io.Unit_Dimension.J: attr[6]
                        }
                else:
                    dest.set_attribute(key, attr)
        container_types = [
            io.Mesh_Container, io.Particle_Container, io.ParticleSpecies,
            io.Record, io.Mesh
        ]
        if isinstance(src, io.Series):
            # main loop: read iterations of src, write to dest
            write_iterations = dest.write_iterations()
            for in_iteration in src.read_iterations():
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
                sys.stdout.flush()
                self.__copy(
                    in_iteration, out_iteration,
                    current_path + str(in_iteration.iteration_index) + "/")
                in_iteration.close()
                out_iteration.close()
                self.chunks.clear()
                sys.stdout.flush()
        elif isinstance(src, io.Record_Component):
            shape = src.shape
            offset = [0 for _ in shape]
            dtype = src.dtype
            dest.reset_dataset(io.Dataset(dtype, shape))
            if src.empty:
                pass # empty record component automatically created by
                     # dest.reset_dataset()
            elif src.constant:
                dest.make_constant(src.get_attribute("value"))
            else:
                chunk = Chunk(offset, shape)
                local_chunk = chunk.slice1D(self.comm.rank, self.comm.size)
                if debug:
                    end = local_chunk.offset.copy()
                    for i in range(len(end)):
                        end[i] += local_chunk.extent[i]
                    print("{}\t{}/{}:\t{} -- {}".format(
                        current_path, self.comm.rank, self.comm.size,
                        local_chunk.offset, end))
                chunk = src.load_chunk(local_chunk.offset, local_chunk.extent)
                self.chunks.append(chunk)
                dest.store_chunk(chunk, local_chunk.offset, local_chunk.extent)
        elif isinstance(src, io.Iteration):
            self.__copy(src.meshes, dest.meshes, current_path + "meshes/")
            self.__copy(src.particles, dest.particles,
                        current_path + "particles/")
        elif any([
                isinstance(src, container_type)
                for container_type in container_types
        ]):
            for key in src:
                self.__copy(src[key], dest[key], current_path + key + "/")


if __name__ == "__main__":
    args = parse_args()
    pipe = pipe(args.infile, args.outfile, args.inconfig, args.outconfig,
                MPI.COMM_WORLD)
    pipe.run()

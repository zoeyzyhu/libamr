#ifndef SRC_MESHLIB_HPP_ 
#define SRC_MESHLIB_HPP_

// C/C++
#include <vector>
#include <array>
#include <memory>

using Real = double;

/// \note This is the staggering of a coordinate axis
/// + : cell center
/// | : cell face
/// ----|-----+------|-----+------|----+------|-----+------|-----+------|---
/// ----|--x1v(i-2)--|--x1v(i-1)--|--x1v(i)---|--x1v(i+1)--|--x1v(i+2)--|---
/// ----|-----+------|-----+------|----+------|-----+------|-----+------|---
/// x1f(i-2)--+--x1f(i-1)--+--x1f(i)---+--x2f(i+i)--+--x1f(i+2)--+--x1f(i+3)
/// ----|-----+------|-----+------|----+------|-----+------|-----+------|---

enum {
  NGHOST = 3
};

using GridPositionIndicator = std::array<int, 3>;

class Coordinate {
 public:
  //! \todo implement default constructor
  Coordinate();

  //! \todo implement constructor
  //! \param[in] pmb pointer to a mesh block
  Coordinate(MeshBlock const *pmb);

  //! virtual destructor (tobe inherited)
  virtual ~Coordinate();

  //! cell center coordinats
  std::vector<Real> x1v, x2v, x3v;

  //! cell face coordinates
  std::vector<Real> x1f, x2f, x3f;
};

using CoordinatePtr = std::shared_ptr<Coordinate>;

//! \brief A MeshBlock is a collection of cells
class MeshBlock {
 public:  // public data
   //! pointer to a coordinate object
   CoordinatePtr coord;
   
   //! opaque class
   class Phy;

   //! opaque pointer to implement physics
   std::shared_ptr<Phy> phy;

 public:
   //! \todo implement default constructor
   MeshBlock();

   //! \todo implement constructor
   //! \param[in] nx1 number of internal cells in x1 direction
   //! \param[in] nx2 number of internal cells in x2 direction
   //! \param[in] nx3 number of internal cells in x3 direction
   MeshBlock(int nx1, int nx2, int nx3);

   //! number of cells in each direction
   int nc1, nc2, nc3;

   //! start and end indices of the direction
   int is, ie, js, je, ks, ke;
};

using MeshBlockPtr = std::shared_ptr<MeshBlock>;

//! \brief A Mesh is a collection of MeshBlocks
class Mesh {
 public:
  //! \todo implement default constructor
  Mesh();

  //! container for all mesh blocks within a mesh
  std::vector<std::shared_ptr<MeshBlock>> blocks;
};

using MeshPtr = std::shared_ptr<Mesh>;

//! \brief Factory class to create a mesh
class MeshFactory {
 public:
  MeshPtr CreateUniformMesh(int nx1, int nx2, int nx3);
};

#endif  // SRC_MESHLIB_HPP_

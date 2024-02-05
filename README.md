## Motivation

Adaptive mesh refinement (AMR) has emerged as a popular and powerful method in scientific research across various disciplines due to its ability to enhance the accuracy and efficiency of numerical simulations. In fluid dynamics, astrophysics, materials science, computer science and engineering, where simulations involve a wide range of length scales and complex interactions, AMR allows for a dynamic and targeted allocation of computational resources. Unlike uniform mesh methods, AMR selectively refines the mesh in regions of interest, enabling high-resolution representation where intricate features or rapid changes occur, while coarsening the mesh in less critical areas. Adaptability optimizes computational efficiency, enabling researchers to focus computational resources where they matter most, thereby reducing overall simulation costs.

Traditional approaches often utilize MPI for parallelism, which is well-suited for uniform mesh methods but can be challenging to implement and scale for AMR. The dynamic nature of AMR, with its irregular mesh structure and varying computational workloads, requires a more flexible and scalable parallel programming model.

In this primitive project, we implement a scalable ARM framework using Ray, a distributed computing library that provides a simple and flexible API for building and scaling parallel and distributed applications. We demonstrate the potential of Ray for scientific computing by distributing AMR simulations on a Slurm-based cluster.

## Design and Implementation

### Tree Data Structure

We employ a tree data structure to efficiently locate mesh blocks within the AMR simulation. The root node spans the entire coordinate system range of the block, and during the refinement process, a new set of leaf nodes is generated from each parent node. The number of leaf nodes is determined by 2 to the power of the dimension. To enhance mesh block localization, each TreeNode carries essential information known as `LogicLocation`. This information captures the sequence of all child nodes relative to the parent node.

The `LogicLocation` is designed using a zigzag order, employing binary indicators (0 and 1) in each dimension. These indicators serve to distinguish different directions, such as left or right in the x-direction. For example, in the x-direction, the binary indicators dictate whether a child node is positioned to the left or right. This approach makes LogicLocation a unique ID for blocks, allowing for precise navigation within the tree structure.

In cases where a child node is a leaf node (having no further children), it represents an actor (mesh block) and is responsible for storing information about its neighboring mesh blocks. This strategic organization ensures an efficient and structured representation of mesh blocks within the simulation.

### MeshBlock

In our MeshBlock implementation for Adaptive Mesh Refinement (AMR), we follow a high-level design that revolves around representing a mesh block within a three-dimensional space. The key component is a class that encapsulates critical information about the block's size, including its dimensions, coordinate type, and the presence of ghost zones. This class efficiently allocates memory for the mesh block, creating views for both interior and ghost zones.

### Actor Model

In our approach to parallelizing AMR simulations using Ray, each mesh block is represented by an “actor”, a distributed, stateful object that encapsulates the data, geographic information, a list of handles of neighboring actors, and methods of a specific mesh block within the AMR simulation. This actor-based model allows for a natural representation of the dynamic nature of mesh refinement and de-refinement, with each actor managing its own data points and coordinates within the mesh. The actor model aligns well with the spatial adaptability required in AMR simulations, enabling a scalable and intuitive parallelization strategy.

For our specific use case in the AMR simulation, we prioritize resource adequacy for each actor and, thus, set "num_cpus=1." Although this setting may result in occasional underutilization of computational resources, the workload distribution remains relatively balanced.

In future implementations, we envision a dynamic adjustment of the number of CPUs based on the refinement level. Upon actor launch, we propose assigning 0.1 CPUs to the most coarsened block, 0.2 CPUs to a block at refinement level 2, and a whole CPU to blocks at level 10.

## Future Work

This project is an exploration of Ray's capability in handling scientific computing workloads as a course project. Due to the time limit, we have only implemented a basic AMR framework using Ray in Python. In the future, we plan to extend this work in the following ways:

- There are plenty of opportunities in optimizing the performance of Ray based on further understanding during the project. For example, in our current setup, each actor performs stencil calculations by looping through the elements of its data block. To enhance computational efficiency, a potential improvement involves dividing the data block into smaller chunks and launching a batch of remote tasks to concurrently execute the stencil calculations. Leveraging Ray's preference for data locality in task scheduling, these tasks would likely be assigned to the same node but different CPUs, striking a balance between communication costs and computational improvements.

- Another important aspect of potential optimization is the refinement process. Currently, each block reports to the manager whether it should refine or coarse post-calculation. To prevent conflicts in refining or merging blocks, the manager serially instructs each block to split, avoiding potential overlapping operations. A potential enhancement involves exploring parallelization with the introduction of locks to safeguard common neighbors during concurrent operations. Alternatively, the refinement process could be divided into two stages. In the first stage, the manager updates the tree structure, launches new actors in need, without considering their requirements to fill interior data. In the second stage, the manager is aware of a non-conflicting list of blocks, and processes them in parallel to fill their interior matrix data through prolongation or restriction operations.

- An intriguing avenue for future exploration is the integration of temporal adaptation into our AMR model. Ray's dynamic scheduling mechanism offers an opportunity to allocate resources proportionally based on the computational needs of the simulation at different time steps. This adaptive approach can potentially optimize the trade-off between accuracy and efficiency, allowing the model to dynamically adjust its temporal resolution in response to varying simulation conditions. This represents a significant step toward a more comprehensive and responsive AMR framework.

- To solidify the scientific computing capabilities of Ray, another future plan is to implement the ARM using C++ and Ray's C++ API. This implementation will allow us to leverage the performance benefits of C++ and explore the potential of Ray in handling complex scientific computing workloads in a more efficient manner.

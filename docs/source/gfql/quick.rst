.. _gfql-quick:

GFQL Quick Reference
====================

This quick reference page provides short examples of various parameters and usage patterns.

Basic Usage
-----------

**Chaining Operations**

.. code-block:: python

    g.chain(ops=[...], engine=EngineAbstract.AUTO)

:meth:`chain <graphistry.compute.chain>` sequences multiple matchers for more complex patterns of paths and subgraphs

- **ops**: Sequence of graph node and edge matchers (:class:`ASTObject <graphistry.compute.ast.ASTObject>` instances).
- **engine**: Optional execution engine. Engine is typically not set, defaulting to `'auto'`. Use `'cudf'` for GPU acceleration and `'pandas'` for CPU.

Node Matchers
-------------

.. code-block:: python

  n(filter_dict=None, name=None, query=None)

:meth:`n <graphistry.compute.ast.n>` matches nodes based on their attributes.

- Filter nodes based on attributes.

- **Parameters**:

  - `filter_dict`: `{attribute: value}` or `{attribute: condition_function}`
  - `name`: Optional label; adds a boolean column in the result.
  - `query`: Custom query string (e.g., `"age > 30 and country == 'USA'"`).

**Examples:**

- Match nodes where `type` is `'person'`:

  .. code-block:: python

      n({"type": "person"})

- Match nodes with `age` greater than 30:

  .. code-block:: python

      n({"age": lambda x: x > 30})

- Use a custom query string:

  .. code-block:: python

      n(query="age > 30 and country == 'USA'")

Edge Matchers
-------------

.. code-block:: python

  e_forward(edge_match=None, hops=1, to_fixed_point=False, source_node_match=None, destination_node_match=None, source_node_query=None, destination_node_query=None, edge_query=None, name=None)
  e_reverse(edge_match=None, hops=1, to_fixed_point=False, source_node_match=None, destination_node_match=None, source_node_query=None, destination_node_query=None, edge_query=None, name=None)
  e_undirected(edge_match=None, hops=1, to_fixed_point=False, source_node_match=None, destination_node_match=None, source_node_query=None, destination_node_query=None, edge_query=None, name=None)
  
  # alias for e_undirected
  e(edge_match=None, hops=1, to_fixed_point=False, source_node_match=None, destination_node_match=None, source_node_query=None, destination_node_query=None, edge_query=None, name=None)

:meth:`e <graphistry.compute.ast.e>` matches edges based on their attributes (undirected). May also include matching on edge's source and destination nodes.

- Traverse edges in the forward direction.

- **Parameters**:

  - `edge_match`: `{attribute: value}` or `{attribute: condition_function}`
  - `edge_query`: Custom query string for edge attributes.
  - `hops`: `int`, number of hops to traverse.
  - `to_fixed_point`: `bool`, continue traversal until no more matches.
  - `source_node_match`: Filter for source nodes.
  - `destination_node_match`: Filter for destination nodes.
  - `source_node_query`: Custom query string for source nodes.
  - `destination_node_query`: Custom query string for destination nodes.
  - `name`: Optional label.

**Examples:**

- Traverse 2 hops forward on edges where `status` is `'active'`:

  .. code-block:: python

      e_forward({"status": "active"}, hops=2)

- Use custom edge query strings:

  .. code-block:: python

      e_forward(edge_query="weight > 5 and type == 'connects'")

- Filter source and destination nodes with match dictionaries:

  .. code-block:: python

      e_forward(
          source_node_match={"status": "active"},
          destination_node_match={"age": lambda x: x < 30}
      )

- Filter source and destination nodes with queries:

  .. code-block:: python

      e_forward(
          source_node_query="status == 'active'",
          destination_node_query="age < 30"
      )

- Label matched edges:

  .. code-block:: python

      e_forward(name="active_edges")

:class:`e_reverse <graphistry.compute.ast.e_reverse>`, :class:`e_forward <graphistry.compute.ast.e_forward>`, and :class:`e <graphistry.compute.ast.e>` are aliases.

- :class:`e_reverse <graphistry.compute.ast.e_reverse>`: Same as :class:`e_forward <graphistry.compute.ast.e_forward>`, but traverses in reverse.
- :class:`e <graphistry.compute.ast.e>`: Traverses edges regardless of direction.

Predicates
-----------

:class:`graphistry.compute.predicates.ASTPredicate.ASTPredicate`

- Matches using a predicate on entity attributes.

See :doc:`predicates/quick` for more information.

**Example:**

- Match nodes where `category` is `'A'`, `'B'`, or `'C'`:

  .. code-block:: python

      from graphistry import n, is_in

      n({"category": is_in(["A", "B", "C"])})

Combined Examples
-----------------

- **Find people connected to transactions via active relationships:**

  .. code-block:: python

      g.chain([
          n({"type": "person"}),
          e_forward({"status": "active"}),
          n({"type": "transaction"})
      ])

- **Label nodes and edges during traversal:**

  .. code-block:: python

      g.chain([
          n({"id": "start_node"}, name="start"),
          e_forward(name="edge1"),
          n({"level": 2}, name="middle"),
          e_forward(name="edge2"),
          n({"type": "end_type"}, name="end")
      ])

- **Traverse until no more matches (fixed point):**

  .. code-block:: python

      g.chain([
          n({"status": "infected"}),
          e_forward(to_fixed_point=True),
          n(name="reachable")
      ])

- **Filter by multiple conditions:**

  .. code-block:: python

      g.chain([
          n({"type": is_in(["server", "database"])}),
          e_undirected({"protocol": "TCP"}, hops=3),
          n(query="risk_level >= 8")
      ])

- **Use custom queries in matchers:**

  .. code-block:: python

      g.chain([
          n(query="age > 30 and country == 'USA'"),
          e_forward(edge_query="weight > 5"),
          n(query="status == 'active'")
      ])

GPU Acceleration
----------------

- **Enable GPU mode:**

  .. code-block:: python

      g.chain([...], engine='cudf')

- **Example with cuDF DataFrames:**

  .. code-block:: python

      import cudf

      e_gdf = cudf.from_pandas(edge_df)
      n_gdf = cudf.from_pandas(node_df)

      g = graphistry.nodes(n_gdf, 'node_id').edges(e_gdf, 'src', 'dst')
      g.chain([...], engine='cudf')

Remote Mode
-----------

- **Query existing remote data**

  .. code-block:: python

      g = graphistry.bind(dataset_id='ds-abc-123')

      nodes_df = g.chain_remote([n()])._nodes

- **Upload graph and run GFQL**

  .. code-block:: python

      g2 = g1.upload()

      g3 = g2.chain_remote([n(), e(), n()])

- **Enforce CPU and GPU mode on remote GFQL**

  .. code-block:: python

      g3a = g2.chain_remote([n(), e(), n()], engine='pandas') 
      g3b = g2.chain_remote([n(), e(), n()], engine='cudf')

- **Return only nodes and certain columns**

  .. code-block:: python

      cols = ['id', 'name']
      g2b = g1.chain_remote([n(), e(), n()], output_type="edges", edge_col_subset=cols)

- **Return only edges and certain columns**

  .. code-block:: python

      cols = ['src', 'dst']
      g2b = g1.chain_remote([n(), e(), n()], output_type="edges", edge_col_subset=cols)

- **Return only shape metadata**

  .. code-block:: python

      shape_df = g1.chain_remote_shape([n(), e(), n()])

- **Run remote Python and get back a graph**

  .. code-block:: python

      def my_remote_trim_graph_task(g):
          return (g
              .nodes(g._nodes[:10])
              .edges(g._edges[:10])
          )

      g2 = g1.upload()
      g3 = g2.python_remote_g(my_remote_trim_graph_task)

- **Run remote Python and get back a table**

  .. code-block:: python

      def first_n_edges(g):
          return g._edges[:10]

      some_edges_df = g.python_remote_table(first_n_edges)

- **Run remote Python and get back JSON**

  .. code-block:: python

      def first_n_edges(g):
          return g._edges[:10].to_json()

      some_edges_json = g.python_remote_json(first_n_edges)

- **Run remote Python and ensure runs on CPU or GPU**

  .. code-block:: python

      g3a = g2.python_remote_g(my_remote_trim_graph_task, engine='pandas')
      g3b = g2.python_remote_g(my_remote_trim_graph_task, engine='cudf')

- **Run remote Python, passing as a string**

  .. code-block:: python

      g2 = g1.upload()

      # ensure method is called "task" and takes a single argument "g"
      g3 = g2.chain_remote_python("""
          def task(g):
              return (g
                  .nodes(g._nodes[:10])
                  .edges(g._edges[:10])
              )
          my_remote_trim_graph_task(g)
      """)

Advanced Usage
--------------

- **Traversal with source and destination node filters and queries:**

  .. code-block:: python

      e_forward(
          edge_query="type == 'follows' and weight > 2",
          source_node_match={"status": "active"},
          destination_node_query="age < 30",
          hops=2,
          name="social_edges"
      )

- **Node matcher with all parameters:**

  .. code-block:: python

      n(
          filter_dict={"department": "sales"},
          query="age > 25 and tenure > 2",
          name="experienced_sales"
      )

- **Edge matcher with all parameters:**

  .. code-block:: python

      e_reverse(
          edge_match={"transaction_type": "refund"},
          edge_query="amount > 100",
          source_node_match={"status": "inactive"},
          destination_node_match={"region": "EMEA"},
          name="large_refunds"
      )

Parameter Summary
-----------------

- **Common Parameters:**

  - `filter_dict`: Attribute filters (e.g., `{"status": "active"}`)
  - `query`: Custom query string (e.g., `"age > 30"`)
  - `hops`: Number of steps to traverse (`int`, default `1`)
  - `to_fixed_point`: Continue traversal until no more matches (`bool`, default `False`)
  - `name`: Label for matchers (`str`)
  - `source_node_match`, `destination_node_match`: Filters for connected nodes
  - `source_node_query`, `destination_node_query`: Queries for connected nodes
  - `edge_match`: Filters for edges
  - `edge_query`: Query for edges
  - `engine`: Execution engine (`EngineAbstract.AUTO`, `'cudf'`, etc.)

Traversal Directions
--------------------

- **Forward Traversal:** `e_forward(...)`
- **Reverse Traversal:** `e_reverse(...)`
- **Undirected Traversal:** `e_undirected(...)`

Tips and Best Practices
-----------------------

- **Limit hops for performance:** Specify `hops` to control traversal depth.
- **Use naming for analysis:** Apply `name` to label and filter results.
- **Combine filters:** Use `filter_dict` and `query` for precise matching.
- **Leverage GPU acceleration:** Use `engine='cudf'` for large datasets.
- **Avoid infinite loops:** Be cautious with `to_fixed_point=True` in cyclic graphs.

Examples at a Glance
--------------------

- **Find all paths between two nodes:**

  .. code-block:: python

      g.chain([
          n({g._node: "Alice"}),
          e_undirected(hops=3),
          n({g._node: "Bob"})
      ])

- **Match nodes with IDs in a range:**

  .. code-block:: python

      n(query="100 <= id <= 200")

- **Traverse edges with specific labels:**

  .. code-block:: python

      e_forward({"label": is_in(["knows", "likes"])})

- **Identify subgraphs based on attributes:**

  .. code-block:: python

      g.chain([
          n({"community": "A"}),
          e_undirected(hops=2),
          n({"community": "B"}, name="bridge_nodes")
      ])

- **Custom edge and node queries:**

  .. code-block:: python

      g.chain([
          n(query="age >= 18"),
          e_forward(edge_query="interaction == 'message'"),
          n(query="location == 'NYC'")
      ])


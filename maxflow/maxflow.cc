#include "maxflow-v3.01.src/graph.h"
#include "maxflow-v3.01.src/graph.cpp"
#include "maxflow-v3.01.src/maxflow.cpp"

#include <vector>

std::vector<double> MakePairwisePotentialsDiagonal(
    double *unary, const int n_unaries, double *pairwise,
    int n_pairwise_cliques, int *edges, const int n_edges) {
  /* E is 3xn_edges array, G is n_pairwise_cliquesx2x2 array, Q is 2xn_unaries
   * array
   * return lambda is nG array
   */

  const double eps = 2e-15;
  std::vector<double> qa, qb, qc, qd;
  std::vector<double> lambda(n_pairwise_cliques);

  for (int k = 0; k < n_pairwise_cliques; ++k) {
    double *g = pairwise + 4 * k;
    // g is C-ordered 2 x 2 matrix written
    double a = 0;
    double b = g[0] - a;
    double c = (g[3] + g[2] - g[1] - g[0]) * 0.5;
    double d = g[3] - c;
    double g11 = g[0] - a - b;
    double g12 = g[1] - a - d;
    double g21 = g[2] - c - b;
    double g22 = g[3] - c - d;
    assert(fabs(g11) < eps && fabs(g22) < eps && fabs(g12 - g21) < eps);
    lambda[k] = (g12 + g21) / 2;
    // update unary potentials
    qa.push_back(a);
    qb.push_back(b);
    qc.push_back(c);
    qd.push_back(d);
  }

  for (int i = 0; i < n_edges; ++i) {
    const int e_type = edges[n_edges * 2 + i];
    double q1[2] = {qa[e_type], qc[e_type]};
    double q2[2] = {qb[e_type], qd[e_type]};
    int u = edges[i];
    int v = edges[i + n_edges];
    for (int k = 0; k < 2; ++k) unary[k * n_unaries + u] += q1[k];
    for (int k = 0; k < 2; ++k) unary[k * n_unaries + v] += q2[k];
  }

  return lambda;
}

// this is old implementation, do not really understand what is happening here
std::vector<double> MakePairwisePotentialsDiagonal2(
    double *unary, const int n_unaries, double *pairwise,
    int n_pairwise_cliques, int *edges, const int n_edges) {
  /* E is 3xn_edges array, G is n_pairwise_cliquesx2x2 array, Q is 2xn_unaries
   * array
   * return lambda is nG array
   */

  const double A_inv[4][4] = {{0, 0.5, 0, 0.5},
                              {-1.0, 0.5, 1.0, 0.5},
                              {1.0, -0.5, 0, -0.5},
                              {0, 0.5, 0, -0.5}};

  const double eps = 2e-15;
  std::vector<double> lambda(n_pairwise_cliques);
  double x[4] = {0, 0, 0, 0};
  std::vector<double> q1[2], q2[2];

  for (int k = 0; k < n_pairwise_cliques; ++k) {
    double *g = pairwise + 4 * k;
    lambda[k] = (g[1] + g[2] - g[0] - g[3]) / 2;

    g[1] -= lambda[k];
    g[2] -= lambda[k];

    for (int i = 0; i < 4; ++i) {
      x[i] = 0;
      for (int j = 0; j < 4; ++j) x[i] += A_inv[i][j] * g[j];
    }

    g[0] = g[3] = 0;
    g[1] = g[2] = lambda[k];

    for (int i = 0; i < 2; ++i) {
      q1[i].push_back(x[i]);
      q2[i].push_back(x[i + 2]);
    }
  }

  for (int e = 0; e < n_edges; ++e) {
    int eid = edges[n_edges * 2 + e];
    int v1 = edges[e];
    int v2 = edges[e + n_edges];
    for (int i = 0; i < 2; ++i) {
      unary[i * n_unaries + v1] += q1[i][eid];
      unary[i * n_unaries + v2] += q2[i][eid];
    }
  }

  return lambda;
}

/* edges: 3 x n_edges
 * unary: 2 x n_unaries
 * pairwise: n_pairwise_cliques x 2 x 2
 */
std::vector<int> Solve2LabelProblem(double *unary, const int n_unaries,
                                    double *pairwise,
                                    const int n_pairwise_cliques, int *edges,
                                    const int n_edges) {
  // first we need to make zeros nondiagonal elements of pairwise function
  std::vector<double> lambda = MakePairwisePotentialsDiagonal(
      unary, n_unaries, pairwise, n_pairwise_cliques, edges, n_edges);

  // constrauct graph for GraphCut
  Graph<double, double, double> graph(
      n_unaries + 2,
      n_unaries + n_edges);  // no. vertices and no.edges in final graph

  graph.add_node(n_unaries + 2);

  for (int i = 0; i < n_edges; ++i) {
    const int u = edges[i];
    const int v = edges[i + n_edges];
    const int e_type = edges[2 * n_edges + i];
    graph.add_edge(edges[u], edges[v], lambda[e_type], lambda[e_type]);
  }

  for (int i = 0; i < n_unaries; ++i) {
    double q = unary[n_unaries + i] - unary[i];
    if (q >= 0)  // source_edges
      graph.add_tweights(i, q, 0);
    else  // sink_edge
      graph.add_tweights(i, 0, -q);
  }

  graph.maxflow(false);

  std::vector<int> labelling(n_unaries);
  for (int i = 0; i < n_unaries; ++i) labelling[i] = graph.what_segment(i);

  return labelling;
}

std::vector<int> solveMultiLabelProblem(const int nK, double *unary,
                                        const int nT, double *pairwise,
                                        const int n_pairwise_cliques,
                                        int *edges, const int n_edges)
/*
 This is direct implementation of Kto2 transfrom. For details see
 "Transforming an arbitrary minsum problem into a binary one" end of 4th
 section.
 */
{
  const int nK2 = nK * nK;
  const int nKm1 = nK - 1;
  const int nVertices = nT * nKm1 + 2;
  const int nEdges = nT * (nKm1 - 1) + n_edges * nKm1;
  const double kInf = 1e50;
  const double kEps = 2e-15;

  // add additional vertecis edges
  // c((t,k),(t,k+1)) = 0     ,          t in T, k = 1..|K|-2
  // c((t,k+1),(t,k)) = INF   ,          t in T, k = 1..|K|-2

  Graph<double, double, double> graph(nVertices + 2, nEdges + nVertices);
  graph.add_node(nVertices + 2);

  // add additional betwwen_variable potentials
  for (int t = 0, offset = 0; t < nT; ++t) {
    for (int k = 0; k + 1 < nKm1; ++k) {
      graph.add_edge(offset + k, offset + k + 1, 0, kInf);
    }
    offset += nKm1;
  }

  std::vector<double> delta_q(nVertices - 2);

  for (int e = 0; e < n_edges; ++e) {
    const int t1 = edges[e], t2 = edges[n_edges + e];
    const double *const g = pairwise + nK2 * edges[n_edges * 2 + e];

    for (int k1 = 0; k1 < nKm1; ++k1) {
      for (int k2 = 0; k2 < nKm1; ++k2) {
        const double alpha = g[k1 * nK + k2] + g[(k1 + 1) * nK + k2 + 1] -
                             g[(k1 + 1) * nK + k2] - g[k1 * nK + k2 + 1];

        if (alpha > kEps) printf("alpha = %f.15\n", alpha);
        assert(alpha <= kEps);

        graph.add_edge(t1 * nKm1 + k1, t2 * nKm1 + k2, -alpha / 2, -alpha / 2);
      }

      delta_q[t1 * nKm1 + k1] += g[k1 * nK] + g[k1 * nK + nKm1] -
                                 g[(k1 + 1) * nK] - g[(k1 + 1) * nK + nKm1];
      delta_q[t2 * nKm1 + k1] +=
          g[k1] + g[nKm1 * nK + k1] - g[k1 + 1] - g[nKm1 * nK + k1 + 1];
    }
  }

  for (int t = 0; t < nT; ++t) {
    for (int k = 0; k + 1 < nK; ++k) {
      double q = unary[nT * k + t] - unary[nT * (k + 1) + t] +
                 delta_q[t * nKm1 + k] / 2;

      if (q >= 0)  // source_edges
        graph.add_tweights(t * nKm1 + k, q, 0);
      else  // sink_edge
        graph.add_tweights(t * nKm1 + k, 0, -q);
    }
  }

  graph.maxflow(false);

  std::vector<int> labelling(nT);

  for (int i = 0; i < nT; ++i) {
    int k = 0;
    for (; k + 1 < nK; ++k)
      if (graph.what_segment(i * (nK - 1) + k)) break;
    labelling[i] = k;
  }

  return labelling;
}
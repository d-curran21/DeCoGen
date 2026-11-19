import os
import random
import shutil
from statistics import mean
import torch
import networkx as nx

from graphgen.train import predict_graphs as gen_graphs_dfscode_rnn
from baselines.graph_rnn.train import predict_graphs as gen_graphs_graph_rnn
from baselines.dgmg.train import predict_graphs as gen_graphs_dgmg
from utils import get_model_attribute, load_graphs, save_graphs

import metrics.stats

LINE_BREAK = '----------------------------------------------------------------------\n'


class ArgsEvaluate():
    def __init__(self):
        # Can manually select the device too
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # === UPDATE THIS TO YOUR MODEL ===
        self.model_path = 'model_save/DFScodeRNN_amazon_directsup_2026-09-14 10:46:39/DFScodeRNN_amazon_directsup_2004.dat'

        self.num_epochs = get_model_attribute('epoch', self.model_path, self.device)

        # Whether to generate networkx graphs from the trained model
        self.generate_graphs = True

        self.count = 2560
        self.batch_size = 32  # Must divide count
        self.metric_eval_batch_size = 40

        # DFScodeRNN
        self.max_num_edges = 102

        # GraphRNN (kept for compatibility if you switch model)
        self.min_num_node = 40
        self.max_num_node = 103

        self.train_args = get_model_attribute('saved_args', self.model_path, self.device)

        self.graphs_save_path = 'graphs/'
        self.current_graphs_save_path = (
            self.graphs_save_path
            + self.train_args.fname + '_' + self.train_args.time + '/'
            + str(self.num_epochs) + '/'
        )


def patch_graph(graph: nx.Graph) -> nx.Graph:
    """Standardize node label field (strip suffix after first '-')."""
    for _, data in graph.nodes(data=True):
        raw_label = data.get('label', '')
        data['label'] = str(raw_label).split('-', 1)[0]
    return graph


def generate_graphs(eval_args):
    """
    Generate graphs (networkx format) from a trained generative model and save them.
    """
    train_args = eval_args.train_args

    if train_args.note == 'GraphRNN':
        gen_graphs = gen_graphs_graph_rnn(eval_args)
    elif train_args.note == 'DFScodeRNN':
        gen_graphs = gen_graphs_dfscode_rnn(eval_args)
    elif train_args.note == 'DGMG':
        gen_graphs = gen_graphs_dgmg(eval_args)
    else:
        raise ValueError(f"Unknown model note: {train_args.note}")

    if os.path.isdir(eval_args.current_graphs_save_path):
        shutil.rmtree(eval_args.current_graphs_save_path)
    os.makedirs(eval_args.current_graphs_save_path)

    save_graphs(eval_args.current_graphs_save_path, gen_graphs)


def print_stats(
    node_count_avg_ref, node_count_avg_pred, edge_count_avg_ref,
    edge_count_avg_pred, degree_mmd, clustering_mmd, orbit_mmd
):
    print('Node count avg: Test - {:.6f}, Generated - {:.6f}'.format(
        mean(node_count_avg_ref), mean(node_count_avg_pred)))
    print('Edge count avg: Test - {:.6f}, Generated - {:.6f}'.format(
        mean(edge_count_avg_ref), mean(edge_count_avg_pred)))

    print('MMD Degree - {:.6f}, MMD Clustering - {:.6f}, MMD Orbits - {:.6f}'.format(
        mean(degree_mmd), mean(clustering_mmd), mean(orbit_mmd)))

    print(LINE_BREAK)


if __name__ == "__main__":
    eval_args = ArgsEvaluate()
    train_args = eval_args.train_args

    print('Evaluating {}, run at {}, epoch {}'.format(
        train_args.fname, train_args.time, eval_args.num_epochs))

    # Generate graphs from the trained model (if requested)
    if eval_args.generate_graphs:
        generate_graphs(eval_args)

    # Split seed: must equal the seed used in main.py when this model was trained
    random.seed(2)

    # Build dataset indices
    graphs = []
    for name in os.listdir(train_args.current_dataset_path):
        if name.endswith('.dat'):
            graphs.append(len(graphs))

    random.shuffle(graphs)
    split = int(0.90 * len(graphs))
    graphs_test_indices = graphs[split:]
    graphs_train_indices = graphs[:split]

    # Generated graph indices
    if not eval_args.generate_graphs:
        graphs_pred_indices = [
            i for i, name in enumerate(os.listdir(eval_args.current_graphs_save_path))
            if name.endswith('.dat')
        ]
    else:
        graphs_pred_indices = [i for i in range(eval_args.count)]

    print('Evaluating {}, run at {}, epoch {}'.format(
        train_args.fname, train_args.time, eval_args.num_epochs))

    print('Graphs generated - {}'.format(len(graphs_pred_indices)))
    print('Test graphs - {}'.format(len(graphs_test_indices)))

    # Running aggregates
    node_count_avg_ref, node_count_avg_pred = [], []
    edge_count_avg_ref, edge_count_avg_pred = [], []
    degree_mmd, clustering_mmd, orbit_mmd = [], [], []

    total_pred = len(graphs_pred_indices)
    for i in range(0, total_pred, eval_args.metric_eval_batch_size):
        batch_cap = min(eval_args.metric_eval_batch_size, total_pred - i)
        if not graphs_test_indices:
            raise RuntimeError("No test graphs found in the dataset split.")
        batch_size = min(batch_cap, len(graphs_test_indices))

        # Reference (real) graphs for this batch
        graphs_ref_indices = random.sample(graphs_test_indices, batch_size)
        graphs_ref = load_graphs(train_args.current_dataset_path, graphs_ref_indices)
        graphs_ref = [patch_graph(g) for g in graphs_ref]

        # Generated graphs for this batch
        graphs_pred = load_graphs(eval_args.current_graphs_save_path, graphs_pred_indices[i: i + batch_size])
        graphs_pred = [patch_graph(g) for g in graphs_pred]

        # Basic metrics
        node_count_avg_ref.append(mean([len(G.nodes()) for G in graphs_ref]))
        node_count_avg_pred.append(mean([len(G.nodes()) for G in graphs_pred]))
        edge_count_avg_ref.append(mean([len(G.edges()) for G in graphs_ref]))
        edge_count_avg_pred.append(mean([len(G.edges()) for G in graphs_pred]))

        # MMDs
        degree_mmd.append(metrics.stats.degree_stats(graphs_ref, graphs_pred))
        clustering_mmd.append(metrics.stats.clustering_stats(graphs_ref, graphs_pred))
        orbit_mmd.append(metrics.stats.orbit_stats_all(graphs_ref, graphs_pred))

        print('Running average of metrics:\n')
        print_stats(
            node_count_avg_ref, node_count_avg_pred, edge_count_avg_ref, edge_count_avg_pred,
            degree_mmd, clustering_mmd, orbit_mmd
        )

    print('Evaluating {}, run at {}, epoch {}'.format(
        train_args.fname, train_args.time, eval_args.num_epochs))
    print('FINAL')
    print_stats(
        node_count_avg_ref, node_count_avg_pred, edge_count_avg_ref, edge_count_avg_pred,
        degree_mmd, clustering_mmd, orbit_mmd
    )

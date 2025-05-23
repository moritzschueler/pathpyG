from manim import *
import pathpyG as pp
from tqdm import tqdm
from matplotlib.pyplot import get_cmap
import matplotlib.colors as mcolors

config.pixel_height = 1080
config.pixel_width = 1960

def get_layout(graph: pp.TemporalGraph, type: str = 'fr', time_window: tuple = None):
    layout_style = {}
    layout_style['layout'] = type
   
    layout = pp.layout(graph.to_static_graph(time_window), **layout_style)
    for key in layout.keys():
        layout[key] = np.append(layout[key], 0.0) # manim works in 3 dimensions, not 2 --> add zeros as third dimension to every node coordinate

    layout_array = np.array(list(layout.values())) 
    mins = layout_array.min(axis=0) # compute the mins and maxs of the 3 dimensions
    maxs = layout_array.max(axis=0)
    center = (mins + maxs) / 2 # compute the center of the network
    scale = 4.0 / (maxs - mins).max() # compute scale, so that every node fits into a 2 x 2 box
    
    for k in layout:
        layout[k] = (layout[k] - center) * scale # scale the position of each node
    
    return layout


class TemporalNetworkPlot(Scene):
    def construct(self):

        g = pp.io.read_csv_temporal_graph('edges_test.csv', is_undirected=False)

        ####### the user specifies these variables in pp.plot(t, ...) ##########

        start = 0 # start time of the simulation
        end = 31 # end time of the simulation
        delta = 1000 # time needed for progressing one time step
        intervals = None # number of numeric intervals, if None --> intervals = num of timesteps (end - start)

        #????
        dynamic_layout_interval = None # specifies after how many time steps a new layout is computed

        #### Keyword Arguments
        node_size = None # dict
        node_color = [('1', 5, 0.3), ('2', 5, 8), ('2', 10, 0.1), ('3', 0, 0.12)]
        node_cmap = get_cmap()
        node_opacity = None # dict
        node_label = None
        
        edge_size = None
        edge_color = [(('1','2'), 0, 0.5), (('2','1'), 2, 0.1)] #['red', 'yellow', 'blue']
        edge_cmap = get_cmap()
        edge_opacity = None

        ########################################################################
        
        # if intervals is not specified, every timestep is an interval
        if intervals == None:
            intervals = end - start

        delta/=1000 # convert milliseconds to seconds

        #colors of nodes
        color_dict = {}
        if isinstance(node_color, str):
            for node in g.nodes:
                color_dict[node] = node_color
        
        elif node_cmap != None and isinstance(node_color, (int, float)):
            node_color = node_cmap(node_color)[:3]
            node_color = (mcolors.to_hex(node_color))
            for node in g.nodes:
                color_dict[node] = node_color

        elif isinstance(node_color, list) and all(isinstance(item, str) for item in node_color):
            for i, node in enumerate(g.nodes):
                color_dict[node] = node_color[i%len(node_color)]

        elif isinstance(node_color, list) and all(isinstance(item, (int, float)) for item in node_color) and node_cmap != None:
            color_list = []
            for color in node_color:
                color_list.append(mcolors.to_hex(node_cmap(color)[:3]))
            node_color = color_list
            for i, node in enumerate(g.nodes):
                color_dict[node] = node_color[i%len(node_color)]

        elif isinstance(node_color, list) and all(isinstance(item, tuple) for item in node_color):
            for node, t, color in node_color:
                if isinstance(color, (int, float)) and node_cmap != None:
                    color = mcolors.to_hex(node_cmap(color)[:3])
                if t == start: # node gets initialized with the right color
                    color_dict[node] = color
                else:
                    color_dict[(node, t)] = color

        #colors of edges
        if isinstance(edge_color, str):
            for edge in g.temporal_edges:
                v,w,t = edge
                edge = v,w
                color_dict[(edge,t)] = edge_color
        
        elif edge_cmap != None and isinstance(edge_color, (int, float)):
            edge_color = edge_cmap(edge_color)[:3]
            edge_color = (mcolors.to_hex(edge_color))
            print(edge_color)
            for edge in g.temporal_edges:
                v,w,t = edge
                edge = v,w
                color_dict[(edge,t)] = edge_color

        elif isinstance(edge_color, list) and all(isinstance(item, str) for item in edge_color):
            for i, temporal_edge in enumerate(g.temporal_edges):
                v, w, t = temporal_edge
                edge = (v,w)
                color_dict[(edge,t)] = edge_color[i%len(edge_color)]

        elif isinstance(edge_color, list) and all(isinstance(item, (int, float)) for item in edge_color) and node_cmap != None:
            color_list = []
            for color in edge_color:
                color_list.append(mcolors.to_hex(node_cmap(color)[:3]))
            edge_color = color_list
            for i, temporal_edge in enumerate(g.temporal_edges):
                v, w, t = temporal_edge
                edge = (v,w)
                color_dict[(edge,t)] = edge_color[i%len(edge_color)]
        
        elif isinstance(edge_color, list) and all(isinstance(item, tuple) for item in edge_color):
            for edge, t, color in edge_color:
                if isinstance(color, (int, float)) and node_cmap != None:
                    color = mcolors.to_hex(node_cmap(color)[:3])
                print(edge)
                color_dict[(edge, t)] = color

        layout = get_layout(g, 'random' if dynamic_layout_interval != None else 'fr')


        time_stamps = g.data['time']
        time_stamps = [timestamp.item() for timestamp in time_stamps]
        time_stamp_dict = dict((time, []) for time in time_stamps)
        for v,w,t in g.temporal_edges:
            time_stamp_dict[t].append((v,w))
        
        graph = Graph(
            g.nodes, [],
            layout=layout,
            labels=False,
            vertex_config={
                v: {"radius": 0.04, "fill_color": color_dict[v] if v in color_dict else BLUE} for v in g.nodes

            }
        )
        self.add(graph)  # create initial nodes
        step_size = int((end - start + 1)/intervals) # step size based on the number of intervals
        time_window = range(start, end+1, step_size)

        change = False
        for time_step in tqdm(time_window):
            range_stop = time_step + step_size
            range_stop = range_stop if range_stop < end + 1 else end + 1

            if step_size == 1 or time_step == end:
                text = Text(f'T = {time_step}')
            else:
                text = Text(f'T = {time_step} to T = {range_stop - 1}')
            text.to_corner(UL)
            self.add(text)


           

            
            for step in range(time_step, range_stop, 1):
                # dynamic layout change
                if dynamic_layout_interval != None and (step - start)%dynamic_layout_interval == 0 and step - start != 0 and change: # change the layout based on the edges since the last change until the current timestep and only if there were edges in the last interval
                    change = False
                    new_layout = get_layout(g, time_window = (step-dynamic_layout_interval, step))

                    animations = []
                    for node in g.nodes:
                        if node in new_layout:
                            new_pos = new_layout[node]
                            animations.append(graph[node].animate.move_to(new_pos))
                    self.play(*animations, run_time=delta)
                
                # color change
                for node in g.nodes:
                    if (node, step) in color_dict:
                        graph[node].set_fill(color_dict[(node, step)])

            lines = []
            for step in range(time_step, range_stop, 1): # generate Lines for all the timesteps in the current interval
                if step in time_stamp_dict:
                    for edge in time_stamp_dict[step]:
                        u, v = edge
                        sender = graph[u].get_center()
                        receiver = graph[v].get_center()
                        line = Line(sender, receiver, stroke_width = 0.4, color = color_dict[(edge, step)] if (edge, step) in color_dict else GRAY)
                        lines.append(line)
            if len(lines) > 0:
                change = True
                self.add(*lines)
                self.wait(delta)
                self.remove(*lines)
            else:
                self.wait(delta)
                    
            self.remove(text)
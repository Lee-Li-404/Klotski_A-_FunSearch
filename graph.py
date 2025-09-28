import json
from collections import defaultdict

class GraphLogger:
    def __init__(self, board):
        self.board = board
        self.id_of = {}
        self.nodes = []
        self.edges = []
        self.parent = {}
        self.root_id = None
        self.goal_id = None
        self.solution_path_ids = []

    def add_or_get(self, s, g, f, parent_state):
        if s in self.id_of:
            return self.id_of[s]
        node_id = len(self.nodes)
        if parent_state is None:
            self.root_id = node_id
            depth = 0
        else:
            depth = self.nodes[self.id_of[parent_state]]["depth"] + 1
        node = {
            "id": node_id,
            "depth": depth,
            "g": g,
            "f": f,
            "in_path": False,
            "pieces": [(p.w, p.h, p.x, p.y) for p in s.pieces],
            "children": []
        }
        self.nodes.append(node)
        self.id_of[s] = node_id
        if parent_state is not None:
            self.edges.append({"source": self.id_of[parent_state], "target": node_id})
            self.parent[s] = parent_state
            self.nodes[self.id_of[parent_state]]["children"].append(node_id)
        else:
            self.parent[s] = None
        return node_id

    def mark_solution_path(self, goal_state):
        cur = goal_state
        ids = []
        while cur is not None:
            nid = self.id_of[cur]
            self.nodes[nid]["in_path"] = True
            ids.append(nid)
            cur = self.parent[cur]
        ids.reverse()
        self.solution_path_ids = ids
        self.goal_id = ids[-1] if ids else None

    def _compute_subtree_sizes(self):
        from collections import defaultdict
        children = defaultdict(list)
        for e in self.edges:
            children[e["source"]].append(e["target"])
        subtree_size = {}

        def dfs(u):
            size = 0
            for v in children.get(u, []):
                size += 1 + dfs(v)
            subtree_size[u] = size
            return size

        if self.root_id is not None:
            dfs(self.root_id)
        for n in self.nodes:
            n["subtree_size"] = subtree_size.get(n["id"], 0)

    def write_html(self, out_path="astar_exploration.html"):
        self._compute_subtree_sizes()

        hidden = {n["id"]: n for n in self.nodes}
        visible_nodes, visible_edges = [], []

        if self.root_id is not None:
            root = hidden.pop(self.root_id)
            visible_nodes.append(root)
            for e in self.edges:
                if e["source"] == self.root_id:
                    child = hidden.pop(e["target"], None)
                    if child:
                        visible_nodes.append(child)
                        visible_edges.append(e)

        data = {
            "board": {"W": self.board.W, "H": self.board.H},
            "visible_nodes": visible_nodes,
            "visible_edges": visible_edges,
            "hidden": hidden,
            "root_id": self.root_id,
            "goal_id": self.goal_id,
            "solution_path_ids": self.solution_path_ids,
            "all_nodes": self.nodes,
            "all_edges": self.edges
        }

        html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>A* Exploration Viewer</title>
<style>
  html, body {{margin:0;padding:0;height:100%;}}
  #app {{display:flex;height:100%;}}
  #graph {{flex:2;}}
  #panel {{flex:1;border-left:1px solid #ccc;padding:8px;}}
  #stateCanvas {{border:1px solid #aaa;}}
</style>
<script src="https://unpkg.com/cytoscape@3.28.1/dist/cytoscape.min.js"></script>
<script>
const DATA = {json.dumps(data)};

document.addEventListener("DOMContentLoaded", function(){{
  const cy = cytoscape({{
    container: document.getElementById('graph'),
    elements: {{
      nodes: DATA.visible_nodes.map(n=>({{data:n, position:{{x:0,y:0}}}})),
      edges: DATA.visible_edges.map(e=>({{data:e}}))
    }},
    layout: {{name:'breadthfirst', directed:true, padding:10, spacingFactor:1.5}},
    style:[
      {{
        selector:'node',
        style:{{
          'label': ele => {{
            let d=ele.data();
            return d.id + " ("+ d.subtree_size + ")";
          }},
          'font-size': 8,
          'text-valign': 'top',
          'color':'#333',
          'background-color': ele=> ele.data('in_path')? '#2b8a3e':'#666',
          'width':10, 'height':10
        }}
      }},
      {{selector:'edge', style:{{'line-color':'#bbb','width':1}}}},
      {{selector:'node:selected', style:{{'background-color':'#d9480f','width':12,'height':12}}}}
    ]
  }});

  const canvas=document.getElementById('stateCanvas');
  const ctx=canvas.getContext('2d');
  function drawState(pieces){{
    const W=DATA.board.W,H=DATA.board.H;
    const cs=Math.floor(380/Math.max(W,H));
    const pad=10; canvas.width=W*cs+pad*2; canvas.height=H*cs+pad*2;
    ctx.fillStyle='#f6f6f6'; ctx.fillRect(0,0,canvas.width,canvas.height);
    ctx.strokeStyle='#ddd';
    for(let i=0;i<=W;i++){{ctx.beginPath();ctx.moveTo(pad+i*cs,pad);ctx.lineTo(pad+i*cs,pad+H*cs);ctx.stroke();}}
    for(let j=0;j<=H;j++){{ctx.beginPath();ctx.moveTo(pad,pad+j*cs);ctx.lineTo(pad+W*cs,pad+j*cs);ctx.stroke();}}
    pieces.forEach((p,idx)=>{{
      const [pw,ph,px,py]=p;
      ctx.fillStyle= idx===0? '#ffcc00':'#88aaff';
      ctx.fillRect(pad+px*cs,pad+py*cs,pw*cs,ph*cs);
      ctx.strokeRect(pad+px*cs,pad+py*cs,pw*cs,ph*cs);
    }});
  }}

  const expanded = new Set(); // store string ids

  // 左键：预览
  cy.on('tap','node', evt => {{
    const d = evt.target.data();
    if(d.pieces && d.pieces.length) drawState(d.pieces);
  }});

  // 右键：展开 / 收起
  cy.on('cxttap','node', evt => {{
    const d = evt.target.data();
    const nid = String(d.id);
    const nodeEle = cy.$id(nid);

    if(expanded.has(nid)){{
      // 收起整棵子树
      const sub = cy.elements().bfs({{ roots: nodeEle, directed: true }}).path;
      const toRemove = sub.not(nodeEle);
      toRemove.nodes().forEach(n => expanded.delete(n.id()));
      cy.remove(toRemove);
      expanded.delete(nid);
    }} else {{
      // 展开
      const child_ids = d.children || [];
      const px = nodeEle.position('x');
      const py = nodeEle.position('y');
      child_ids.forEach((cid,i) => {{
        const child = DATA.hidden[cid];
        if(!child) return;
        child.parent_id = d.id;
        const cx = px + (i - child_ids.length/2) * 30;
        const cyy = py + 80;
        cy.add({{ data:child, position:{{x:cx,y:cyy}} }});
        cy.add({{ data:{{source:d.id, target:child.id}} }});
      }});
      expanded.add(nid);
    }}
  }});

  // 🔥 一键展开解路径
  document.getElementById('expandPath').onclick = () => {{
    const ids = DATA.solution_path_ids || [];
    const nodeMap = {{}};
    DATA.all_nodes.forEach(n => nodeMap[n.id] = n);

    ids.forEach((nid,i) => {{
      const node = nodeMap[nid];
      if(!node) return;
      if(cy.$id(""+nid).nonempty()) return;

      let parent_id = (i > 0) ? ids[i-1] : null;
      let px=0, py=0;
      if(parent_id && cy.$id(""+parent_id).nonempty()){{
        px = cy.$id(""+parent_id).position('x');
        py = cy.$id(""+parent_id).position('y') + 80;
      }}
      node.parent_id = parent_id;
      cy.add({{ data:node, position:{{x:px,y:py}} }});
      if(parent_id) cy.add({{ data:{{source:parent_id, target:nid}} }});
    }});
    cy.fit();
  }};
}});
</script>
</head>
<body>
<div id="app">
  <div id="graph"></div>
  <div id="panel">
    <h3>Preview</h3>
    <canvas id="stateCanvas" width="400" height="500"></canvas>
    <p><button id="expandPath">show solution path</button></p>
  </div>
</div>
</body>
</html>"""
        with open(out_path,"w",encoding="utf-8") as f:
            f.write(html)
        print(f"[A* Viewer] wrote {out_path}")

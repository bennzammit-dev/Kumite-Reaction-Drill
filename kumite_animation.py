import bpy
import random
import math
import os
from mathutils import Vector

# =============================================================================
# CONFIG
# =============================================================================
NUM_STUDENTS = 9       # change per run: 9, 10, 21, 22, 30
AREA_SIZE    = 10.0
MIN_DIST     = 1.5
MAX_ATTEMPTS = 30
FPS          = 24

BELTS = [
    "white", "white_stripe", "yellow", "orange",
    "green", "blue", "blue_stripe",
    "brown", "brown_1_stripe", "brown_2_stripe", "black"
]
BELT_RANK = {b: i for i, b in enumerate(BELTS)}

BODY_IMAGES   = {"M": "male.png", "F": "female.png"}
WORKED_IMAGES = ["nws.png", "ws.png", "hws.png"]

SUBFOLDERS    = ["TImageT", "TImageT/Belts", "TImageT/Bodies", "TImageT/States"]
TEMPLATE_NAME = "StudentTemplate"
AGE_TEXT_NAME = "AgeText"

# Algorithm config (mirrors kumite_evaluation.ipynb)
NWS_MIN = 2
NWS_MAX = 6
MAX_CLUSTER_IMBALANCE = 1
GMM_SIGMA      = MIN_DIST * 2.5
GMM_MIN_WEIGHT = 0.05

GA1_POP       = 50
GA1_GENS      = 200
GA1_CR        = 0.8
GA1_MR        = 0.05
GA1_MS        = 0.5
GA1_TOURN     = 3
GA1_RECOMPUTE = 0.1
GA1_STEP_EVERY= 20      # keyframe every N generations
WS_NWS_TARGET = MIN_DIST * 2.0
W_DIST  = 1.0; W_ANGLE = 1.0; W_CONS = 0.8; W_SIZE = 1.5; W_BAL = 2.0

GA2_POP       = 50
GA2_GENS      = 200
GA2_CR        = 0.8
GA2_MR        = 0.05
GA2_TOURN     = 3
GA2_STEP_EVERY= 20
GA2_NEIGHBOURS= NWS_MIN + 1
W_BELT = 3.0; W_AGE = 1.0; W_GENDER = 0.5

RL_ROLLOUTS = 2000

# =============================================================================
# ANIMATION TIMING
# Pattern per step: hold 0.5s → move arrives at 1.5s → hold 0.5s → next
# So step N has:
#   hold_start  = N * 2.0          seconds  (identical to end of previous)
#   hold_pre    = N * 2.0 + 0.5    seconds  (same position, pre-move freeze)
#   move_arrive = N * 2.0 + 1.5    seconds  (new position keyframe)
#   hold_post   = N * 2.0 + 2.0    seconds  (same as move_arrive, post-move freeze)
# Step 0 is the spawn — no pre-hold needed, just set position and hold.
# =============================================================================

_step_counter = [0]   # mutable so nested functions can increment it

def current_step():
    return _step_counter[0]

def next_step():
    _step_counter[0] += 1
    return _step_counter[0]

def frames_for_step(step):
    """
    Returns (f_hold_start, f_hold_pre, f_move, f_hold_post) for a given step.
    Step 0 is special: f_hold_start == f_hold_pre == 0 (spawn frame).
    """
    t = step * 2.0
    if step == 0:
        return (0, 0, 0, int(0.5 * FPS))
    f_hold_start = int(t * FPS)
    f_hold_pre   = int((t + 0.5) * FPS)
    f_move       = int((t + 1.5) * FPS)
    f_hold_post  = int((t + 2.0) * FPS)
    return (f_hold_start, f_hold_pre, f_move, f_hold_post)

def keyframe_positions(blender_objs_at_positions, step):
    """
    blender_objs_at_positions: list of (blender_obj, (x, y)) pairs
    Applies the 4-frame pattern for this step:
      hold_start  — same as previous step's hold_post (already set externally)
      hold_pre    — identical position, freeze before move
      move        — new position arrives
      hold_post   — identical to move, freeze after
    For step 0 only the position + hold_post are set.
    """
    f_hs, f_hp, f_mv, f_po = frames_for_step(step)

    for obj, (px, py) in blender_objs_at_positions:
        if step == 0:
            obj.location = Vector((px, py, 0.0))
            obj.keyframe_insert(data_path="location", frame=0)
            obj.keyframe_insert(data_path="location", frame=f_po)
        else:
            # Pre-hold: insert at current position (obj.location unchanged yet)
            obj.keyframe_insert(data_path="location", frame=f_hs)
            obj.keyframe_insert(data_path="location", frame=f_hp)
            # Move arrives
            obj.location = Vector((px, py, 0.0))
            obj.keyframe_insert(data_path="location", frame=f_mv)
            obj.keyframe_insert(data_path="location", frame=f_po)

def keyframe_role(obj, genome, new_role, step):
    """
    Updates the WorkedMat colour and keyframes it at the move and hold-post frames.
    Uses Base Color on Principled BSDF (or Color on Emission) since Blender
    keyframes shader node inputs natively — unlike image node image assignments.
    Colours:
      0 = NWS  -> red    (1.0, 0.0, 0.0, 1.0)
      1 = WS   -> orange (1.0, 0.5, 0.0, 1.0)
      2 = HWS  -> green  (0.0, 0.8, 0.1, 1.0)
    """
    ROLE_COLOURS = {
        0: (1.0, 0.0, 0.0, 1.0),
        1: (1.0, 0.5, 0.0, 1.0),
        2: (0.0, 0.8, 0.1, 1.0),
    }
    colour = ROLE_COLOURS[new_role]
    _, _, f_mv, f_po = frames_for_step(step)

    genome["worked"] = new_role
    obj["worked"]    = new_role

    mesh_objs = ([obj] if hasattr(obj, "material_slots") else []) + \
                [o for o in obj.children_recursive if hasattr(o, "material_slots")]

    for mo in mesh_objs:
        for slot in mo.material_slots:
            mat = slot.material
            if not mat or not mat.name.startswith("WorkedMat") or not mat.use_nodes:
                continue
            if mat.node_tree and mat.node_tree.users > 1:
                mat.node_tree = mat.node_tree.copy()

            colour_node = None
            colour_input = "Base Color"
            for node in mat.node_tree.nodes:
                if node.bl_idname == "ShaderNodeBsdfPrincipled":
                    colour_node = node
                    colour_input = "Base Color"
                    break
                if node.bl_idname == "ShaderNodeEmission":
                    colour_node = node
                    colour_input = "Color"

            if colour_node is None:
                print(f"[WARN] No Principled BSDF or Emission in {mat.name}")
                continue

            inp = colour_node.inputs.get(colour_input)
            if inp is None:
                print(f"[WARN] Input not found in {mat.name}")
                continue

            inp.default_value = colour
            inp.keyframe_insert("default_value", frame=f_mv)
            inp.keyframe_insert("default_value", frame=f_po)

    obj.keyframe_insert(data_path='["worked"]', frame=f_mv)
    obj.keyframe_insert(data_path='["worked"]', frame=f_po)

# =============================================================================
# BLENDER HELPERS (from file.py — unchanged)
# =============================================================================

def get_blend_dir():
    blend_path = bpy.data.filepath
    if not blend_path:
        raise RuntimeError("Save the .blend file first.")
    return os.path.dirname(blend_path)

def load_all_images():
    blend_dir = get_blend_dir()
    all_names = list(BODY_IMAGES.values()) + WORKED_IMAGES + [b + ".png" for b in BELTS]
    for name in all_names:
        if name in bpy.data.images:
            continue
        for sub in SUBFOLDERS:
            full_path = os.path.join(blend_dir, sub, name)
            if os.path.exists(full_path):
                try:
                    bpy.data.images.load(full_path)
                    print(f"[IMG] {sub}/{name}")
                except Exception as e:
                    print(f"[ERR] {full_path}: {e}")
                break

def copy_materials(obj):
    for slot in obj.material_slots:
        if slot.material:
            slot.material = slot.material.copy()

def make_materials_unique(obj):
    for slot in obj.material_slots:
        mat = slot.material
        if mat is None:
            continue
        if mat.users > 1:
            slot.material = mat.copy()
            mat = slot.material
        if mat.use_nodes and mat.node_tree and mat.node_tree.users > 1:
            mat.node_tree = mat.node_tree.copy()

def assign_images_to_materials(obj, genome):
    """
    Assigns body and belt images via texture nodes (unchanged from file.py).
    WorkedMat colour is set via Base Color / Emission Color — not an image —
    so that it can be properly keyframed during the animation.
    """
    ROLE_COLOURS = {
        0: (1.0, 0.0, 0.0, 1.0),
        1: (1.0, 0.5, 0.0, 1.0),
        2: (0.0, 0.8, 0.1, 1.0),
    }
    for slot in obj.material_slots:
        mat = slot.material
        if not mat or not mat.use_nodes:
            continue
        mn = mat.name

        if mn.startswith("WorkedMat"):
            # Set colour directly on the shader node — keyframeable
            colour = ROLE_COLOURS.get(genome["worked"], ROLE_COLOURS[0])
            for node in mat.node_tree.nodes:
                if node.bl_idname == "ShaderNodeBsdfPrincipled":
                    node.inputs["Base Color"].default_value = colour
                    break
                if node.bl_idname == "ShaderNodeEmission":
                    node.inputs["Color"].default_value = colour
                    break
            continue

        # Body and Belt — image texture assignment (same as file.py)
        img_node = None
        for node in mat.node_tree.nodes:
            if node.bl_idname == "ShaderNodeTexImage":
                if mn.startswith("BodyMat") and getattr(node,"label","")=="IMG_Body":
                    img_node = node; break
                elif mn.startswith("BeltMat") and getattr(node,"label","")=="IMG_Belt":
                    img_node = node; break
                elif img_node is None:
                    img_node = node
        if not img_node:
            continue
        if mn.startswith("BodyMat"):
            img = bpy.data.images.get(BODY_IMAGES.get(genome["gender"]))
        elif mn.startswith("BeltMat"):
            img = bpy.data.images.get(genome["belt"] + ".png")
        else:
            img = None
        if img:
            img_node.image = img

def copy_object_with_children(template, new_name):
    dup_map = {}
    def dup_obj(orig):
        new = orig.copy()
        if orig.data:
            try: new.data = orig.data.copy()
            except: new.data = orig.data
        new.name = f"{orig.name}_{new_name}"
        dup_map[orig] = new
        bpy.context.collection.objects.link(new)
        return new
    roots = [template]; all_objs = []
    while roots:
        o = roots.pop(); all_objs.append(o); roots.extend(o.children)
    for o in all_objs:
        dup_obj(o)
    for orig, dup in dup_map.items():
        if orig.parent:
            dup.parent = dup_map.get(orig.parent)
            dup.matrix_parent_inverse = orig.matrix_parent_inverse.copy()
        else:
            dup.parent = None
        dup.location = orig.location.copy()
        dup.rotation_euler = orig.rotation_euler.copy()
        dup.scale = orig.scale.copy()
    return dup_map.get(template)

# =============================================================================
# SPAWN (all students start as NWS — roles assigned after GA1)
# =============================================================================

def spawn_students(positions):
    template = bpy.data.objects.get(TEMPLATE_NAME)
    if template is None:
        raise RuntimeError(f"No object named '{TEMPLATE_NAME}' found.")

    students = []
    for i, (px, py) in enumerate(positions):
        new_root = copy_object_with_children(template, f"{i:03d}")
        if new_root is None:
            print(f"[ERR] Failed to duplicate template for student {i}")
            continue
        new_root.name = f"Student_{i:03d}"

        age_text_obj = None
        for child in new_root.children_recursive:
            if child.type == 'FONT' or child.name.startswith(AGE_TEXT_NAME):
                age_text_obj = child; break

        new_root.location = Vector((px, py, 0.0))

        genome = {
            "gender": random.choice(["M", "F"]),
            "belt"  : random.choice(BELTS),
            "age"   : random.randint(8, 60),
            "worked": 0,   # everyone starts NWS — roles assigned after GA1
        }

        new_root["student_index"] = i
        new_root["gender"] = genome["gender"]
        new_root["belt"]   = genome["belt"]
        new_root["age"]    = genome["age"]
        new_root["worked"] = genome["worked"]

        mesh_objs = ([new_root] if hasattr(new_root,"material_slots") else []) + \
                    [o for o in new_root.children_recursive if hasattr(o,"material_slots")]
        for mo in mesh_objs:
            copy_materials(mo)
            make_materials_unique(mo)
            assign_images_to_materials(mo, genome)

        if age_text_obj and age_text_obj.type == 'FONT' and age_text_obj.data:
            try: age_text_obj.data = age_text_obj.data.copy()
            except: pass
            age_text_obj.data.body = str(genome["age"])

        # Keyframe initial role colour at frame 0 so playback starts correctly
        ROLE_COLOURS = {
            0: (1.0, 0.0, 0.0, 1.0),
            1: (1.0, 0.5, 0.0, 1.0),
            2: (0.0, 0.8, 0.1, 1.0),
        }
        colour = ROLE_COLOURS[genome["worked"]]
        spawn_mesh_objs = ([new_root] if hasattr(new_root,"material_slots") else []) +                           [o for o in new_root.children_recursive if hasattr(o,"material_slots")]
        for mo in spawn_mesh_objs:
            for slot in mo.material_slots:
                mat = slot.material
                if not mat or not mat.name.startswith("WorkedMat") or not mat.use_nodes:
                    continue
                if mat.node_tree and mat.node_tree.users > 1:
                    mat.node_tree = mat.node_tree.copy()
                for node in mat.node_tree.nodes:
                    if node.bl_idname == "ShaderNodeBsdfPrincipled":
                        node.inputs["Base Color"].default_value = colour
                        node.inputs["Base Color"].keyframe_insert("default_value", frame=0)
                        break
                    if node.bl_idname == "ShaderNodeEmission":
                        node.inputs["Color"].default_value = colour
                        node.inputs["Color"].keyframe_insert("default_value", frame=0)
                        break
        new_root.keyframe_insert(data_path='["worked"]', frame=0)

        students.append((new_root, genome))
        print(f"  Student_{i:03d} | {genome['gender']} | {genome['belt']} | age={genome['age']}")

    return students

# =============================================================================
# PURE PYTHON ALGORITHMS (no bpy dependency)
# Copied directly from kumite_evaluation — identical logic
# =============================================================================

def poisson_disk_sample(area_size, min_dist, num_points, max_attempts=30):
    cs = min_dist / math.sqrt(2); grid = {}
    def tg(x,y): return int((x+area_size)/cs), int((y+area_size)/cs)
    def ib(x,y): return -area_size<=x<=area_size and -area_size<=y<=area_size
    def hn(x,y):
        c,r=tg(x,y)
        for dc in range(-2,3):
            for dr in range(-2,3):
                nb=grid.get((c+dc,r+dr))
                if nb and math.dist((x,y),nb)<min_dist: return True
        return False
    x0,y0=random.uniform(-area_size,area_size),random.uniform(-area_size,area_size)
    pts,active=[(x0,y0)],[(x0,y0)]; grid[tg(x0,y0)]=(x0,y0)
    while active and len(pts)<num_points:
        idx=random.randrange(len(active)); base=active[idx]; placed=False
        for _ in range(max_attempts):
            a=random.uniform(0,2*math.pi); r=random.uniform(min_dist,2*min_dist)
            nx,ny=base[0]+r*math.cos(a),base[1]+r*math.sin(a)
            if ib(nx,ny) and not hn(nx,ny):
                pts.append((nx,ny)); active.append((nx,ny))
                grid[tg(nx,ny)]=(nx,ny); placed=True
                if len(pts)>=num_points: break
        if not placed: active.pop(idx)
    if len(pts)<num_points:
        raise ValueError(f"Poisson disk: only placed {len(pts)}/{num_points}")
    return pts[:num_points]

def kmeans(positions, k, iterations=100):
    n=len(positions)
    centroids=[random.choice(positions)]
    for _ in range(k-1):
        dists=[min(math.dist(p,c)**2 for c in centroids) for p in positions]
        total=sum(dists); r=random.uniform(0,total) if total>0 else 0; cum=0.0
        for p,d in zip(positions,dists):
            cum+=d
            if cum>=r: centroids.append(p); break
        else: centroids.append(positions[-1])
    labels=[0]*n
    for _ in range(iterations):
        nl=[min(range(k),key=lambda c,pos=positions[i]:math.dist(pos,centroids[c]))
            for i in range(n)]
        if nl==labels: break
        labels=nl
        nc=[]
        for cid in range(k):
            mb=[positions[i] for i in range(n) if labels[i]==cid]
            nc.append((sum(p[0] for p in mb)/len(mb),sum(p[1] for p in mb)/len(mb))
                      if mb else random.choice(positions))
        centroids=nc
    for _ in range(n*k):
        counts=[labels.count(c) for c in range(k)]
        if max(counts)-min(counts)<=MAX_CLUSTER_IMBALANCE: break
        src=max(range(k),key=lambda c:counts[c])
        dst=min(range(k),key=lambda c:counts[c])
        sm=[i for i in range(n) if labels[i]==src]
        move=min(sm,key=lambda i:math.dist(positions[i],centroids[dst]))
        labels[move]=dst
        for cid in (src,dst):
            mb=[positions[i] for i in range(n) if labels[i]==cid]
            if mb: centroids[cid]=(sum(p[0] for p in mb)/len(mb),
                                   sum(p[1] for p in mb)/len(mb))
    return centroids,labels

def gmm_w(ws_pos, nws_pos, sigma=None):
    if sigma is None: sigma=GMM_SIGMA
    return math.exp(-(math.dist(ws_pos,nws_pos)**2)/(2*sigma**2))

def ws_group_gmm(ws_idx, all_pos, nws_idx):
    wp=all_pos[ws_idx]
    mb=[(i,gmm_w(wp,all_pos[i])) for i in nws_idx
        if gmm_w(wp,all_pos[i])>=GMM_MIN_WEIGHT]
    mb.sort(key=lambda x:x[1],reverse=True)
    return mb[:NWS_MAX]

def ws_candidacy(positions, centroids, labels):
    k=len(centroids); scores=[]
    for i,pos in enumerate(positions):
        d_own=math.dist(pos,centroids[labels[i]])
        others=[math.dist(pos,centroids[c]) for c in range(k) if c!=labels[i]]
        d_other=min(others) if others else 1.0
        scores.append(d_other/(d_own+1e-6))
    return scores

def get_ws_slots(positions, k, iters):
    centroids,labels=kmeans(positions,k,iters)
    scores=ws_candidacy(positions,centroids,labels)
    ws_slots=[]
    for cid in range(k):
        mb=[(i,scores[i]) for i in range(len(positions)) if labels[i]==cid]
        if mb: ws_slots.append(max(mb,key=lambda x:x[1])[0])
    return ws_slots,centroids,labels

def ga1_group_score(ws_slot, all_pos, nws_slots):
    wp=all_pos[ws_slot]; mb=ws_group_gmm(ws_slot,all_pos,nws_slots)
    if not mb: return 1000.0
    wn=[(all_pos[i],w) for i,w in mb]; eff=sum(w for _,w in wn)
    raw_dp=sum(w*(math.dist(wp,p)-WS_NWS_TARGET)**2 for p,w in wn)/max(eff,1e-6)
    dp=raw_dp/(WS_NWS_TARGET**2)
    if len(wn)>=2:
        tw=sum(w for _,w in wn)
        ss=sum(w*math.sin(math.atan2(p[1]-wp[1],p[0]-wp[0])) for p,w in wn)
        cs=sum(w*math.cos(math.atan2(p[1]-wp[1],p[0]-wp[0])) for p,w in wn)
        ap=math.sqrt((ss/tw)**2+(cs/tw)**2)
    else: ap=0.0
    raw_sp=((NWS_MIN-eff)**2 if eff<NWS_MIN else (eff-NWS_MAX)**2 if eff>NWS_MAX else 0.0)
    sp=raw_sp/(NWS_MAX**2)
    return W_DIST*dp+W_ANGLE*ap+W_SIZE*sp

def ga1_fitness(genome, ws_slots, centroids, labels):
    pos=[(genome[i],genome[i+1]) for i in range(0,len(genome),2)]
    n=len(pos); ws_set=set(ws_slots); nws_s=[i for i in range(n) if i not in ws_set]
    scores=[ga1_group_score(ws,pos,nws_s) for ws in ws_slots]
    if not scores: return float("inf")
    mean=sum(scores)/len(scores); var=sum((s-mean)**2 for s in scores)/len(scores)
    counts=[labels.count(c) for c in range(len(centroids))]
    min_c=min(counts) if counts else 0
    balance=W_BAL*sum(max(0,counts[c]-(min_c+MAX_CLUSTER_IMBALANCE))**2
                      for c in range(len(centroids)))
    collision=sum((MIN_DIST-math.dist(pos[a],pos[b]))**2
                  for a in range(n) for b in range(a+1,n)
                  if math.dist(pos[a],pos[b])<MIN_DIST)
    return mean+W_CONS*math.sqrt(var)+balance+W_BAL*collision

def ga1_cx(a,b):
    if random.random()>GA1_CR: return a[:],b[:]
    cut=random.randint(1,len(a)//2-1)*2
    return a[:cut]+b[cut:],b[:cut]+a[cut:]

def ga1_mut(g):
    return [max(-AREA_SIZE,min(AREA_SIZE,v+random.gauss(0,GA1_MS)))
            if random.random()<GA1_MR else v for v in g]

def compat(s_a, s_b):
    belt=abs(BELT_RANK[s_a["belt"]]-BELT_RANK[s_b["belt"]])/10.0
    age=min(abs(s_a["age"]-s_b["age"]),20)/20.0
    gen=0.0 if s_a["gender"]==s_b["gender"] else 1.0
    return W_BELT*belt+W_AGE*age+W_GENDER*gen

def ga2_fitness(perm, students, positions):
    n=len(perm); total=0.0
    for slot_i in range(n):
        s_a=students[perm[slot_i]]; pos_i=positions[slot_i]
        dists=[(math.dist(pos_i,positions[slot_j]),slot_j)
               for slot_j in range(n) if slot_j!=slot_i]
        dists.sort()
        for _,slot_j in dists[:GA2_NEIGHBOURS]:
            total+=compat(s_a,students[perm[slot_j]])
    return total

def ga2_ox(a,b):
    if random.random()>GA2_CR: return a[:],b[:]
    n=len(a); s=random.randint(0,n-2); e=random.randint(s+1,n-1)
    child=[None]*n; child[s:e+1]=a[s:e+1]; ic=set(child[s:e+1])
    fv=[v for v in b if v not in ic]; fp=[i for i in range(n) if child[i] is None]
    for p,v in zip(fp,fv): child[p]=v
    return child,b[:]

def ga2_mut(perm):
    m=perm[:]
    for i in range(len(m)):
        if random.random()<GA2_MR: j=random.randint(0,len(m)-1); m[i],m[j]=m[j],m[i]
    return m

def rmsd(vals):
    if len(vals)<2: return 0.0
    mu=sum(vals)/len(vals)
    return math.sqrt(sum((v-mu)**2 for v in vals)/len(vals))

def greedy_assign(ws_set, nws_set, pd):
    avail=list(nws_set)
    ws_order=sorted(ws_set,
                    key=lambda w:min(math.dist(pd[w],pd[n]) for n in avail)
                    if avail else 0)
    asgn=[]
    for wi in ws_order:
        if not avail: break
        best=min(avail,key=lambda n:math.dist(pd[wi],pd[n]))
        asgn.append((wi,best)); avail.remove(best)
    return asgn

def run_sequence(ws_start, nws_start, pos_snap, epsilon=0.0):
    ew=set(ws_start); en=set(nws_start); ep_pos=dict(pos_snap)
    assignments=[]; costs=[]
    while ew:
        if en:
            if random.random()<epsilon:
                el=list(en); random.shuffle(el)
                asgn=[(w,n) for w,n in zip(list(ew),el[:len(ew)])]
            else:
                asgn=greedy_assign(ew,en,ep_pos)
            cost=sum(math.dist(ep_pos[w],ep_pos[n]) for w,n in asgn)
            for w,n in asgn: ep_pos[w],ep_pos[n]=ep_pos[n],ep_pos[w]
            used={n for _,n in asgn}; ew=used; en-=used
        else:
            asgn=[]; cost=0.0; ew=set()
        assignments.append(asgn); costs.append(cost)
    return assignments,costs

# =============================================================================
# ANIMATION PIPELINE
# =============================================================================

def run_pipeline(students):
    """
    Runs GA1 → GA2 → RL on the student list and produces Blender keyframes.
    students: list of (blender_obj, genome_dict) — genome has gender/belt/age/worked.
    """
    n = len(students)
    k = math.ceil(n / (NWS_MIN + 1))
    recompute_every = max(1, int(GA1_RECOMPUTE * GA1_GENS))

    # index maps
    obj_of   = {i: students[i][0] for i in range(n)}   # student_index -> blender_obj
    genome_of= {i: students[i][1] for i in range(n)}   # student_index -> genome dict

    # ── STEP 0: spawn positions ──────────────────────────────────────────────
    spawn_pos = [(students[i][0].location.x, students[i][0].location.y)
                 for i in range(n)]

    # keyframe step 0: hold at spawn positions
    keyframe_positions([(obj_of[i], spawn_pos[i]) for i in range(n)], step=0)
    step = next_step()

    print("Running GA1...")
    seed_g = [v for p in spawn_pos for v in p]
    pop = [[max(-AREA_SIZE,min(AREA_SIZE,v+random.gauss(0,GA1_MS))) for v in seed_g]
           for _ in range(GA1_POP)]

    cached_ws, cached_c, cached_l = get_ws_slots(spawn_pos, k, 20)
    best_score_ever = float("inf")
    best_genome_ever = seed_g[:]

    for gen in range(GA1_GENS):
        if gen > 0 and gen % recompute_every == 0:
            bg = min(pop, key=lambda g: ga1_fitness(g, cached_ws, cached_c, cached_l))
            bp = [(bg[i],bg[i+1]) for i in range(0,len(bg),2)]
            cached_ws, cached_c, cached_l = get_ws_slots(bp, k, 20)

        fit = lambda g: ga1_fitness(g, cached_ws, cached_c, cached_l)
        best = min(pop, key=fit)
        score = fit(best)

        if score < best_score_ever:
            best_score_ever = score
            best_genome_ever = best[:]

        new_pop = [best[:]]
        while len(new_pop) < GA1_POP:
            pa = min(random.sample(pop,GA1_TOURN),key=fit)
            pb = min(random.sample(pop,GA1_TOURN),key=fit)
            ca,cb = ga1_cx(pa,pb)
            new_pop.append(ga1_mut(ca))
            if len(new_pop)<GA1_POP: new_pop.append(ga1_mut(cb))
        pop = new_pop

        # Keyframe every GA1_STEP_EVERY generations — show current best
        if (gen+1) % GA1_STEP_EVERY == 0:
            bp = [(best[i],best[i+1]) for i in range(0,len(best),2)]
            # slot i = student i at this point (no permutation yet)
            keyframe_positions([(obj_of[i], bp[i]) for i in range(n)], step=step)
            step = next_step()
            print(f"  GA1 gen {gen+1} | fitness {score:.4f} | anim step {step-1}")

    # Final best position (may differ from last plot step)
    bg = best_genome_ever
    ga1_final = [(bg[i],bg[i+1]) for i in range(0,len(bg),2)]
    final_ws_slots, final_c, final_l = get_ws_slots(ga1_final, k, 100)

    # Keyframe GA1 final best (always — even if last interval already showed it)
    keyframe_positions([(obj_of[i], ga1_final[i]) for i in range(n)], step=step)
    step = next_step()
    print(f"  GA1 best keyframed at anim step {step-1}")

    # Assign WS roles — whoever is in a WS slot gets ws.png
    ws_slot_set = set(final_ws_slots)
    for slot_i in range(n):
        role = 1 if slot_i in ws_slot_set else 0
        genome_of[slot_i]["worked"] = role
        keyframe_role(obj_of[slot_i], genome_of[slot_i], role, step-1)
    print(f"  Roles assigned: {sum(1 for i in range(n) if genome_of[i]['worked']==1)} WS")

    # ── GA2 ──────────────────────────────────────────────────────────────────
    print("Running GA2...")
    base = list(range(n))
    pop2 = [random.sample(base,len(base)) for _ in range(GA2_POP)]
    fit2 = lambda p: ga2_fitness(p, [genome_of[i] for i in range(n)], ga1_final)

    best2_ever = None
    best2_score = float("inf")

    for gen in range(GA2_GENS):
        best2 = min(pop2, key=fit2)
        score2 = fit2(best2)
        if score2 < best2_score:
            best2_score = score2
            best2_ever = best2[:]

        new2 = [best2[:]]
        while len(new2)<GA2_POP:
            pa=min(random.sample(pop2,GA2_TOURN),key=fit2)
            pb=min(random.sample(pop2,GA2_TOURN),key=fit2)
            ca,cb=ga2_ox(pa,pb); new2.append(ga2_mut(ca))
            if len(new2)<GA2_POP: new2.append(ga2_mut(cb))
        pop2=new2

        if (gen+1) % GA2_STEP_EVERY == 0:
            bp2 = best2_ever[:]
            # bp2[slot] = student index — move the object for that student
            # to that slot's position. If role differs from what student
            # currently shows, also update role image.
            pairs = [(obj_of[bp2[slot]], ga1_final[slot]) for slot in range(n)]
            keyframe_positions(pairs, step=step)

            # Role update: slot in ws_slot_set → that student is WS now
            for slot in range(n):
                stu_idx = bp2[slot]
                new_role = 1 if slot in ws_slot_set else 0
                if genome_of[stu_idx]["worked"] != new_role:
                    keyframe_role(obj_of[stu_idx], genome_of[stu_idx], new_role, step)

            step = next_step()
            print(f"  GA2 gen {gen+1} | fitness {score2:.4f} | anim step {step-1}")

    # GA2 final best
    best_perm = best2_ever[:]
    pairs = [(obj_of[best_perm[slot]], ga1_final[slot]) for slot in range(n)]
    keyframe_positions(pairs, step=step)
    for slot in range(n):
        stu_idx = best_perm[slot]
        new_role = 1 if slot in ws_slot_set else 0
        if genome_of[stu_idx]["worked"] != new_role:
            keyframe_role(obj_of[stu_idx], genome_of[stu_idx], new_role, step)
    step = next_step()
    print(f"  GA2 best keyframed at anim step {step-1}")

    # Build final position dict for RL: student_index -> (x,y)
    pos_dict = {best_perm[slot]: ga1_final[slot] for slot in range(n)}
    # Sync roles from final permutation
    for slot in range(n):
        stu_idx = best_perm[slot]
        genome_of[stu_idx]["worked"] = 1 if slot in ws_slot_set else 0

    # ── RL ───────────────────────────────────────────────────────────────────
    print("Running RL rollouts...")
    ws_set  = {i for i in range(n) if genome_of[i]["worked"]==1}
    nws_set = {i for i in range(n) if genome_of[i]["worked"]==0}

    best_total = float("inf")
    best_asgns = None
    best_costs  = None

    for rollout in range(RL_ROLLOUTS):
        eps = 1.0 - (rollout/RL_ROLLOUTS)*0.95
        asgns,costs = run_sequence(ws_set, nws_set, pos_dict, epsilon=eps)
        total = sum(costs)
        if total < best_total:
            best_total = total
            best_asgns = asgns
            best_costs = costs

    print(f"  RL best sequence: total movement={best_total:.3f} "
          f"| RMSD={rmsd([c for c in best_costs if c>0]):.4f}")

    # Reset roles for animation playback
    for i in range(n):
        r = 1 if i in ws_set else 0
        genome_of[i]["worked"] = r
        keyframe_role(obj_of[i], genome_of[i], r, step-1)

    # Replay best sequence with keyframes
    active_ws  = set(ws_set)
    active_nws = set(nws_set)

    for ep_idx, (asgn, cost) in enumerate(zip(best_asgns, best_costs)):
        if asgn:
            # All swaps in this episode happen simultaneously — one anim step
            pairs = []
            for w_idx, n_idx in asgn:
                # both students move to each other's current position
                pos_w = pos_dict[w_idx]
                pos_n = pos_dict[n_idx]
                pairs.append((obj_of[w_idx], pos_n))
                pairs.append((obj_of[n_idx], pos_w))

            keyframe_positions(pairs, step=step)

            # Commit position and role changes
            for w_idx, n_idx in asgn:
                pos_dict[w_idx], pos_dict[n_idx] = pos_dict[n_idx], pos_dict[w_idx]
                keyframe_role(obj_of[w_idx], genome_of[w_idx], 2, step)  # WS -> HWS
                keyframe_role(obj_of[n_idx], genome_of[n_idx], 1, step)  # NWS -> WS

            unmatched = active_ws - {w for w,_ in asgn}
            for w_idx in unmatched:
                keyframe_role(obj_of[w_idx], genome_of[w_idx], 2, step)  # WS -> HWS

            active_ws  = {n for _,n in asgn}
            active_nws -= active_ws
            step = next_step()
            print(f"  RL ep {ep_idx+1}: {len(asgn)} swap(s) | "
                  f"movement={cost:.2f} | anim step {step-1}")
        else:
            # Final episode: remaining WS -> HWS, no movement
            for w_idx in active_ws:
                keyframe_role(obj_of[w_idx], genome_of[w_idx], 2, step)
            active_ws = set()
            step = next_step()
            print(f"  RL ep {ep_idx+1}: final — all WS -> HWS | anim step {step-1}")

    # Set scene end frame
    _, _, _, f_po = frames_for_step(step)
    bpy.context.scene.frame_end = f_po
    print(f"Animation complete. Total steps: {step} | End frame: {f_po}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    bpy.context.scene.render.fps = FPS
    _step_counter[0] = 0   # reset in case script is re-run

    print("Loading images...")
    try:
        load_all_images()
    except RuntimeError as e:
        print(f"[ERR] {e}"); return

    print(f"Poisson disk sampling ({NUM_STUDENTS} students)...")
    try:
        positions = poisson_disk_sample(AREA_SIZE, MIN_DIST, NUM_STUDENTS, MAX_ATTEMPTS)
    except ValueError as e:
        print(f"[ERR] {e}"); return

    print("Spawning students...")
    students = spawn_students(positions)
    if not students:
        print("[ERR] No students spawned."); return

    for img in bpy.data.images:
        if img.users == 0:
            img.use_fake_user = True

    print("Running pipeline and generating animation...")
    run_pipeline(students)
    print("Done.")

if __name__ == "__main__":
    main()

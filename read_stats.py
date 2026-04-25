import json

with open('/tmp/sys_stats.json') as f:
    s = json.load(f)

print("=== SYSTEM ===")
for k in ['total_nodes','occupied_nodes','bridge_count','pulse_count','current_phase']:
    print(f"  {k}: {s.get(k)}")

lc = s.get('lifecycle',{}).get('counts',{})
print(f"  lifecycle: {lc}")
sc = s.get('self_check',{})
print(f"  self_check: {sc.get('total_anomalies_found')} anomalies, {sc.get('total_repairs_done')} repairs")
hb = s.get('hebbian',{})
print(f"  hebbian: {hb.get('total_path_segments')} segs, {hb.get('golden_paths')} golden")
cr = s.get('crystallized',{})
print(f"  crystal: {cr.get('total_crystals')}, transmissions={cr.get('total_transmissions')}")
rg = s.get('self_regulation',{})
print(f"  stress: {rg.get('stress',{}).get('level',0):.3f}, dopamine: {rg.get('hormones',{}).get('dopamine',0):.2f}")
so = s.get('self_organize',{})
print(f"  clusters: {so.get('active_clusters')}, shortcuts: {so.get('active_shortcuts')}")

em = s.get('emergence_summary', {})
print("\n=== EMERGENCE ===")
print(f"  overall: {em.get('overall_score',0):.3f} ({em.get('emergence_level','?')})")
for dim in ['clustering','bridges','crystal','phase']:
    d = em.get(dim, {})
    print(f"  {dim}: {d.get('score',0):.3f}")

with open('/tmp/dp_stats.json') as f:
    dp = json.load(f)

print("\n=== DARK PLANE ENGINE ===")
for k in ['temperature','avg_well_depth','std_well_depth','avg_free_energy','avg_entropy','flow_cycles','total_transitions','total_reawakenings','descent_count','ascent_count']:
    print(f"  {k}: {dp.get(k)}")
print(f"  plane_dist: {dp.get('plane_distribution')}")

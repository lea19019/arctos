import os, torch, numpy as np
os.environ["HF_HUB_OFFLINE"]="1"; torch.set_num_threads(1)
from transformers import AutoTokenizer, AutoModelForCausalLM
MODEL="Qwen/Qwen3-1.7B"
tk=AutoTokenizer.from_pretrained(MODEL); m=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float32).eval()
conds={
"EN_SG":["The doctor near the cabinets","The author of the books","The key to the cabinets","The senator by the trees",
 "The pilot with the maps","The farmer beside the barns","The teacher next to the desks","The lawyer behind the doors",
 "The soldier among the trees","The child near the windows","The engineer with the tools","The dancer beside the mirrors"],
"EN_PL":["The doctors near the cabinet","The authors of the book","The keys to the cabinet","The senators by the tree",
 "The pilots with the map","The farmers beside the barn","The teachers next to the desk","The lawyers behind the door",
 "The soldiers among the tree","The children near the window","The engineers with the tool","The dancers beside the mirror"],
"FR_SG":["Le médecin près des armoires","L'auteur des livres","La clé des armoires","Le sénateur près des arbres",
 "Le pilote avec les cartes","Le fermier près des granges","Le professeur à côté des bureaux","L'avocat derrière les portes",
 "Le soldat parmi les arbres","L'enfant près des fenêtres","L'ingénieur avec les outils","La danseuse près des miroirs"],
"FR_PL":["Les médecins près de l'armoire","Les auteurs du livre","Les clés de l'armoire","Les sénateurs près de l'arbre",
 "Les pilotes avec la carte","Les fermiers près de la grange","Les professeurs à côté du bureau","Les avocats derrière la porte",
 "Les soldats parmi l'arbre","Les enfants près de la fenêtre","Les ingénieurs avec l'outil","Les danseuses près du miroir"],
"TR_SG":["Dolapların yanındaki doktor","Kitapların yazarı","Dolapların anahtarı","Ağaçların yanındaki senatör",
 "Haritaları olan pilot","Ambarların yanındaki çiftçi","Masaların yanındaki öğretmen","Kapıların arkasındaki avukat",
 "Ağaçların arasındaki asker","Pencerelerin yanındaki çocuk","Aletleri olan mühendis","Aynaların yanındaki dansçı"],
"TR_PL":["Dolabın yanındaki doktorlar","Kitabın yazarları","Dolabın anahtarları","Ağacın yanındaki senatörler",
 "Haritası olan pilotlar","Ambarın yanındaki çiftçiler","Masanın yanındaki öğretmenler","Kapının arkasındaki avukatlar",
 "Ağacın arasındaki askerler","Pencerenin yanındaki çocuklar","Aleti olan mühendisler","Aynanın yanındaki dansçılar"],
}
@torch.no_grad()
def per_item(prefixes):
    out=[]
    for p in prefixes:
        lg=m(tk(p,return_tensors="pt").input_ids).logits[0,-1].float()
        out.append(torch.softmax(lg,-1).numpy())
    return np.stack(out)
D={k:per_item(v) for k,v in conds.items()}
print("fwd done", {k:v.shape for k,v in D.items()})
np.save("/tmp/D.npy", D, allow_pickle=True)
def jsd(a,b):
    mm=0.5*(a+b); eps=1e-30
    return float(0.5*np.sum(a*np.log2((a+eps)/(mm+eps)))+0.5*np.sum(b*np.log2((b+eps)/(mm+eps))))
def mean_dist(arr,idx): return arr[idx].mean(0)
n=12; rng=np.random.default_rng(0)
def stat(idx):
    P={k:mean_dist(v,idx) for k,v in D.items()}
    r={}
    r["EN_within_lang"]=jsd(P["EN_SG"],P["EN_PL"]); r["FR_within_lang"]=jsd(P["FR_SG"],P["FR_PL"]); r["TR_within_lang"]=jsd(P["TR_SG"],P["TR_PL"])
    for x,y in [("EN","FR"),("EN","TR"),("FR","TR")]:
        w=0.5*(jsd(P[x+"_SG"],P[y+"_SG"])+jsd(P[x+"_PL"],P[y+"_PL"]))
        b=0.5*(jsd(P[x+"_SG"],P[y+"_PL"])+jsd(P[x+"_PL"],P[y+"_SG"]))
        r[f"{x}{y}_within"]=w; r[f"{x}{y}_between"]=b; r[f"{x}{y}_SIGNAL"]=b-w
    return r
point=stat(np.arange(n))
boots=[stat(rng.integers(0,n,n)) for _ in range(2000)]
print(f"\n{'metric':22s} {'point':>9s} {'95% CI':>22s}")
for k in point:
    v=np.array([b[k] for b in boots]); lo,hi=np.percentile(v,[2.5,97.5])
    print(f"{k:22s} {point[k]:9.4f}   [{lo:7.4f}, {hi:7.4f}]")
print("\nn_items=12 per condition, bootstrap over items, 2000 resamples. JSD in bits, ceiling 1.0")

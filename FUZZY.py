import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

suhu = ctrl.Antecedent(np.arange(0,101,1),'suhu')
ac = ctrl.Consequent(np.arange(0,101,1),'ac')

suhu['dingin'] = fuzz.trimf(suhu.universe,[0,0,50])
suhu['sejuk'] = fuzz.trimf(suhu.universe,[0,50,100])
suhu['normal'] = fuzz.trimf(suhu.universe,[50,100,100])
suhu['hangat'] = fuzz.trimf(suhu.universe,[50,100,150])
suhu['panas'] = fuzz.trimf(suhu.universe,[100,150,150])

ac['rendah'] = fuzz.trimf(ac.universe, [0,0,50])
ac['sedang'] = fuzz.trimf(ac.universe, [0,50,100])
ac['tinggi'] = fuzz.trimf(ac.universe, [50,100,100])

aturan1 = ctrl.Rule(suhu['dingin'], ac['tinggi'])
aturan2 = ctrl.Rule(suhu['sejuk'], ac['sedang'])
aturan3 = ctrl.Rule(suhu['normal'], ac['sedang'])
aturan4 = ctrl.Rule(suhu['hangat'], ac['rendah'])
aturan5 = ctrl.Rule(suhu['panas'], ac['rendah'])

sistem_kontrol = ctrl.ControlSystem([aturan1,aturan2,aturan3,aturan4,aturan5])
pengendali = ctrl.ControlSystemSimulation(sistem_kontrol)

pengendali.input['suhu'] = 75
pengendali.compute()

hasil_panegaturan_ac = pengendali.output['ac']

print(f'Hasil pengaturan AC: {hasil_panegaturan_ac:.2f}')
from tkinter import Label, Tk, Button, Entry
import pickle
import xgboost as xgb
import pandas as pd


root = Tk()
root.title('Heart Failure Prediction')

root.iconbitmap('heart.ico')

label1 = Label(root, text = 'Enter the Age of the patient:', justify = 'left')
label1.pack()

a = Entry(root, width = 50, borderwidth = 3)
a.pack()


label2 = Label(root, text = 'Enter 1 if the patient has a decrease of hemoglobin or 0 otherwise:', justify = 'left')
label2.pack()

b = Entry(root, width = 50, borderwidth = 3)
b.pack()


label3 = Label(root, text = 'Enter the level of the CPK enzyme in the blood (mcg/L):', justify = 'left')
label3.pack()

c = Entry(root, width = 50, borderwidth = 3)
c.pack()

label4 = Label(root, text = 'Enter 1 if the patient has diabetes or 0 otherwise:', justify = 'left')
label4.pack()

d = Entry(root, width = 50, borderwidth = 3)
d.pack()


label5 = Label(root, text = 'Percentage of blood leaving the heart at each contraction (%):', justify = 'left')
label5.pack()

e = Entry(root, width = 50, borderwidth = 3)
e.pack()


label6 = Label(root, text = 'Enter 1 if the patient has hypertension or 0 otherwise::', justify = 'left')
label6.pack()

f = Entry(root, width = 50, borderwidth = 3)
f.pack()


label7 = Label(root, text = 'Enter the level of platelets in the blood (kiloplatelets/mL):', justify = 'left')
label7.pack()

g = Entry(root, width = 50, borderwidth = 3, justify = 'left')
g.pack()


label8 = Label(root, text = 'Enter the level of serum creatinine in the blood (mg/dL):', justify = 'left')
label8.pack()

h = Entry(root, width = 50, borderwidth = 3)
h.pack()


label9 = Label(root, text = 'Enter the level of serum sodium in the blood (mEq/L):', justify = 'left')
label9.pack()

i = Entry(root, width = 50, borderwidth = 3)
i.pack()


label10 = Label(root, text = 'Enter 1 if the patient is a man or 0 if it is a woman:', justify = 'left')
label10.pack()

j = Entry(root, width = 50, borderwidth = 3)
j.pack()


label11 = Label(root, text = 'Enter 1 if the patient is a smoker or 0 otherwise:', justify = 'left')
label11.pack()

k = Entry(root, width = 50, borderwidth = 3)
k.pack()


label12 = Label(root, text = 'Enter the number of days since the patient has had heart failure:', justify = 'left')
label12.pack()

l = Entry(root, width = 50, borderwidth = 3)
l.pack()


label13 = Label(root, text = 'Name of the patient:', justify = 'left')
label13.pack()

m = Entry(root, width = 50, borderwidth = 3)
m.pack()


label14 = Label(root, text = 'ID number of the patient:', justify = 'left')
label14.pack()

n = Entry(root, width = 50, borderwidth = 3)
n.pack()


label15 = Label(root, text = 'Date (dd/mm/yy):', justify = 'left')
label15.pack()

o = Entry(root, width = 50, borderwidth = 3)
o.pack()

label16 = Label(root, text = '', justify = 'left')
label16.pack()

label17 = Label(root, text = '', justify = 'left')
label17.pack()

def predict_outcome():
    
    try:
        ag = float(a.get()),
        bg = float(b.get()),
        cg = float(c.get()),
        dg = float(d.get()),
        eg = float(e.get()),
        fg = float(f.get()),
        gg = float(g.get()),
        hg = float(h.get()),
        ig = float(i.get()),
        jg = float(j.get()),
        kg = float(k.get()),
        lg = float(l.get()),
        mg = str(m.get()),
        ng = int(n.get()),
        og = str(o.get())
        
    except ValueError:
        label16.config(text = '''Review the answers in each entry! Make sure that the numbers
with decimal plates have a. (dot) instead of a , (comma).
Also make sure that there aren't any entries with letters/numbers mixed
letters/special characters mixed, numbers/special characters mixed, except for the Date.''')
        
    new_data_point = [ag, bg, cg, dg, eg, fg, gg, hg, ig, jg, kg, lg, mg, ng, og]
    
    df = pd.DataFrame([], columns = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
                                     'ejection_fraction', 'high_blood_pressure', 'platelets',
                                     'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time',
                                     'patient_name', 'Id_number_patient', 'Date'])
    
    z = 0
    for column in df.columns:
        df[column] = new_data_point[z]
        z += 1
      
    filename = 'xgboost_heart_failure_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.predict(xgb.DMatrix(df[['age', 'anaemia', 'creatinine_phosphokinase',
                                                  'diabetes', 'ejection_fraction',
                                                  'high_blood_pressure', 'platelets',
                                                  'serum_creatinine', 'serum_sodium', 'sex',
                                                  'smoking', 'time']]))
    if result == 1.0:
        label17.config(text = 'The patient is likely to have heart failure!')
    else:
        label17.config(text = 'The patient is unlikely to have heart failure!')
        
def delete():
    a.delete(0, 'end')
    b.delete(0, 'end')
    c.delete(0, 'end')
    d.delete(0, 'end')
    e.delete(0, 'end')
    f.delete(0, 'end')
    g.delete(0, 'end')
    h.delete(0, 'end')
    i.delete(0, 'end')
    j.delete(0, 'end')
    k.delete(0, 'end')
    l.delete(0, 'end')
    m.delete(0, 'end')
    n.delete(0, 'end')
    o.delete(0, 'end')
    
SubmitButton = Button(root, text = 'Submit Values', command = predict_outcome)
SubmitButton.pack()

DeleteEntries = Button(root, text = "Delete all entries", command = delete)
DeleteEntries.pack()

root.mainloop()



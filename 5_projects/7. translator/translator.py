from googletrans import Translator
import tkinter as tk

translator = Translator()


def translate() :
    text = input_text.get("1.0", tk.END).strip()
    target_lang = target_lang_var.get()
    if text:
        translated_text = translator.translate(text, dest=target_lang)
        output_text. config(state="normal")
        output_text.delete("1.0", tk.END)
        output_text.insert(tk.END, translated_text.text)
        output_text. config(state="disabled")

# Create GUI Window
root = tk. Tk()
root.title("Language Translator")
root.geometry("400x300")

# Input Text Zone
input_text = tk.Text(root, height=5, width=40)
input_text.pack(pady=10)

# Target Language Dropdown
target_lang_var = tk.StringVar(root)
target_lang_var.set("es") # Default: Spanish
lang_dropdown = tk.OptionMenu(root, target_lang_var, "es", "fr", "de", "zh", "ar", "ru")
lang_dropdown.pack()

#Button de traduction
translate_button = tk.Button(root, text="Translate", command=translate)
translate_button.pack(pady=5)

# Output Text Area
output_text = tk. Text(root, height=5, width=40, state="disabled")
output_text.pack(pady=10)

root.mainloop()




def translate_text():
    text = input("Entrez le text a traduire")
    src_lang = input("Entre la langue source du text (ex: fr, en, de,...) : ")
    tgt_lang = input("Entre la langue cible du text (ex: fr, en, de, es,...) : ")

    translated_text = translator.translate(text, src=src_lang, dest=tgt_lang)
    print(f"Text traduit : ", translated_text)

translate_text()
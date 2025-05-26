import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import time
import csv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.ticker as ticker
import pandas as pd
import Algorithm as alg

# --- Основной класс GUI ---
class SubgradientGUI:
    def __init__(self, master):
        self.master = master
        master.title("Анализ Субградиентного Метода")

        master.columnconfigure(0, weight=1) 
        master.rowconfigure(0, weight=1)     
        master.columnconfigure(1, weight=1)
        master.rowconfigure(1, weight=1)     
        master.columnconfigure(2, weight=1) 
        master.rowconfigure(2, weight=1)     

        # --- Параметры задачи ---
        self.frame_params = ttk.LabelFrame(master, text="Параметры Задачи")
        self.frame_params.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        for i in range(3):
            self.frame_params.columnconfigure(i, weight=1) 

        ttk.Label(self.frame_params, text="Размер задачи (n):").grid(row=0, column=0, pady=5, padx=5, sticky="e")
        self.n_entry = ttk.Entry(self.frame_params)
        self.n_entry.grid(row=0, column=1, pady=5, padx=5, sticky="ew")

        ttk.Label(self.frame_params, text="Число ограничений (K):").grid(row=1, column=0, pady=5, padx=5, sticky="e")
        self.k_entry = ttk.Entry(self.frame_params)
        self.k_entry.grid(row=1, column=1, pady=5, padx=5, sticky="ew")

        # Кнопки для загрузки данных И автозаполнения
        ttk.Button(self.frame_params, text="Загрузить C", command=self.load_C).grid(row=2, column=0, pady=5, padx=5,  sticky="ew")
        ttk.Button(self.frame_params, text="Сгенерировать C", command=self.generate_C).grid(row=2, column=1, pady=5, padx=5,  sticky="ew")

        ttk.Button(self.frame_params, text="Загрузить D", command=self.load_D).grid(row=3, column=0, pady=5, padx=5,  sticky="ew")
        ttk.Button(self.frame_params, text="Сгенерировать D", command=self.generate_D).grid(row=3, column=1, pady=5, padx=5,  sticky="ew")

        ttk.Button(self.frame_params, text="Загрузить b", command=self.load_b).grid(row=4, column=0, pady=5, padx=5,  sticky="ew")
        ttk.Button(self.frame_params, text="Сгенерировать b", command=self.generate_b).grid(row=4, column=1, pady=5, padx=5,  sticky="ew")


        # --- Параметры алгоритма ---
        self.frame_algo = ttk.LabelFrame(master, text="Параметры Алгоритма")
        self.frame_algo.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        for i in range(3):
            self.frame_algo.columnconfigure(i, weight=1) 

        ttk.Label(self.frame_algo, text="Макс. итераций (N_max):").grid(row=0, column=0, pady=5, padx=5, sticky="e")
        self.n_max_entry = ttk.Entry(self.frame_algo)
        self.n_max_entry.grid(row=0, column=1, pady=5, padx=5, sticky="ew")

        ttk.Label(self.frame_algo, text="Порог точности (ε):").grid(row=1, column=0, pady=5, padx=5, sticky="e")
        self.epsilon_entry = ttk.Entry(self.frame_algo)
        self.epsilon_entry.grid(row=1, column=1, pady=5, padx=5, sticky="ew")

        ttk.Label(self.frame_algo, text="Правило шага:").grid(row=2, column=0, pady=5, padx=5, sticky="e")
        self.step_rule_combo = ttk.Combobox(self.frame_algo, values=["1/n", "const"])
        self.step_rule_combo.grid(row=2, column=1, pady=5, padx=5, sticky="ew")
        self.step_rule_combo.set("1/n")  # Значение по умолчанию

        ttk.Label(self.frame_algo, text="Начальные λ₀ (опционально):").grid(row=3, column=0, pady=5, padx=5, sticky="e")
        self.lambda0_entry = ttk.Entry(self.frame_algo)
        self.lambda0_entry.grid(row=3, column=1, pady=5, padx=5, sticky="ew")

        ttk.Label(self.frame_params, text="Тип задачи:").grid(row=7, column=0, pady=5, padx=5, sticky="e")
        self.ztype_combo = ttk.Combobox(self.frame_params, values=['min', 'max'])
        self.ztype_combo.grid(row=7, column=1, pady=5, padx=5, sticky="ew")
        self.ztype_combo.set('min')  # Значение по умолчанию

        # --- Поле для общего времени ---
        self.total_time_label = ttk.Label(self.frame_params, text="Общее время:")
        self.total_time_label.grid(row=8, column=0, sticky="e", padx=5, pady=5)

        self.total_time_value = tk.StringVar()
        self.total_time_value.set("0.00")  # Начальное значение
        self.total_time_entry = ttk.Entry(self.frame_params, textvariable=self.total_time_value, state="readonly")
        self.total_time_entry.grid(row=8, column=1, sticky="nsew", padx=5, pady=5)

        # Кнопки для экспорта данных
        ttk.Button(self.frame_params, text="Экспортировать C", command=self.export_C).grid(row=2, column=2, pady=5, padx=5, sticky="ew")
        ttk.Button(self.frame_params, text="Экспортировать b", command=self.export_b).grid(row=3, column=2, pady=5, padx=5,  sticky="ew")
        ttk.Button(self.frame_params, text="Экспортировать D", command=self.export_D).grid(row=4, column=2, pady=5, padx=5,  sticky="ew")

        # --- Кнопки управления ---
        self.frame_controls = ttk.Frame(master)
        self.frame_controls.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky="ew")

        for i in range(3):
            self.frame_controls.columnconfigure(i, weight=1) 

        self.start_button = ttk.Button(self.frame_controls, text="Запуск", command=self.start_calculation)
        self.start_button.grid(row=0, column=0, padx=5, sticky="ew")

        self.stop_button = ttk.Button(self.frame_controls, text="Остановка", command=self.stop_calculation, state="disabled")
        self.stop_button.grid(row=0, column=1, padx=5, sticky="ew")

        self.save_log_button = ttk.Button(self.frame_controls, text="Сохранить лог", command=self.save_log)
        self.save_log_button.grid(row=0, column=2, padx=5, sticky="ew")

        # --- Границы генерации ---
        self.frame_generation = ttk.Frame(master)
        self.frame_generation.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")

        for i in range(2):
            self.frame_generation.columnconfigure(i, weight=1)

        # Границы для C
        self.frame_generation_C = ttk.LabelFrame(self.frame_generation, text="Границы для C")
        self.frame_generation_C.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        for i in range(2):
            self.frame_generation_C.columnconfigure(i, weight=1)

        ttk.Label(self.frame_generation_C, text="Нижняя граница:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.lower_bound_C_entry = ttk.Entry(self.frame_generation_C)
        self.lower_bound_C_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.lower_bound_C_entry.insert(0, "0.0")

        ttk.Label(self.frame_generation_C, text="Верхняя граница:").grid(row=1, column=0, padx=5, pady=5,sticky="e")
        self.upper_bound_C_entry = ttk.Entry(self.frame_generation_C)
        self.upper_bound_C_entry.grid(row=1, column=1,padx=5, pady=5, sticky="ew")
        self.upper_bound_C_entry.insert(0, "1.0")

        # Границы для b
        self.frame_generation_b = ttk.LabelFrame(self.frame_generation, text="Границы для b")
        self.frame_generation_b.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

        for i in range(2):
            self.frame_generation_b.columnconfigure(i, weight=1)

        ttk.Label(self.frame_generation_b, text="Нижняя граница:").grid(row=0, column=0,padx=5, pady=5, sticky="e")
        self.lower_bound_b_entry = ttk.Entry(self.frame_generation_b)
        self.lower_bound_b_entry.grid(row=0, column=1,padx=5, pady=5, sticky="ew")
        self.lower_bound_b_entry.insert(0, "0.0")

        ttk.Label(self.frame_generation_b, text="Верхняя граница:").grid(row=1, column=0, padx=5, pady=5,sticky="e")
        self.upper_bound_b_entry = ttk.Entry(self.frame_generation_b)
        self.upper_bound_b_entry.grid(row=1, column=1,padx=5, pady=5, sticky="ew")
        self.upper_bound_b_entry.insert(0, "1.0")

        # Границы для D
        self.frame_generation_D = ttk.LabelFrame(self.frame_generation, text="Границы для D")
        self.frame_generation_D.grid(row=2, column=0, padx=5, pady=5, sticky="ew")

        for i in range(2):
            self.frame_generation_D.columnconfigure(i, weight=1)

        ttk.Label(self.frame_generation_D, text="Нижняя граница:").grid(row=0, column=0,padx=5, pady=5, sticky="e")
        self.lower_bound_D_entry = ttk.Entry(self.frame_generation_D)
        self.lower_bound_D_entry.grid(row=0, column=1,padx=5, pady=5, sticky="ew")
        self.lower_bound_D_entry.insert(0, "0.0")

        ttk.Label(self.frame_generation_D, text="Верхняя граница:").grid(row=1, column=0, sticky="e")
        self.upper_bound_D_entry = ttk.Entry(self.frame_generation_D)
        self.upper_bound_D_entry.grid(row=1, column=1, padx=5, pady=5,sticky="ew")
        self.upper_bound_D_entry.insert(0, "1.0")

        # --- Вывод результатов ---
        self.notebook = ttk.Notebook(master)
        self.notebook.grid(row=2, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")

        # Таблица X
        self.table_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.table_frame, text="Матрица X")
        self.table_frame.rowconfigure(index=0, weight=1)
        self.table_frame.columnconfigure(index=0, weight=1)

        # Создаем Treeview
        self.table = ttk.Treeview(self.table_frame)
        self.table.grid(row=0, column=0, sticky="nsew")

        yscrollbar = ttk.Scrollbar(self.table_frame, orient="vertical", command=self.table.yview)
        self.table.configure(yscroll=yscrollbar.set)
        yscrollbar.grid(row=0, column=1, sticky="ns")

        xscrollbar = ttk.Scrollbar(self.table_frame, orient="horizontal", command=self.table.xview)
        self.table.configure(xscroll=xscrollbar.set)
        xscrollbar.grid(row=1,columnspan=2, column=0, sticky="ew")

        # График нарушения
        self.plot_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.plot_frame, text="График Нарушения")
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Журнал
        self.log_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.log_frame, text="Журнал")
        self.log_text = tk.Text(self.log_frame, wrap="word", font =('Helvetica', 16))
        self.log_frame.rowconfigure(index=0, weight=1)
        self.log_frame.columnconfigure(index=0, weight=1)
        self.log_text.grid(row=0,column=0, sticky="nsew")
        self.scrollbar = ttk.Scrollbar(self.log_frame, command=self.log_text.yview)
        self.scrollbar.grid(row=0,column=1, sticky="nse")
        self.log_text['yscrollcommand'] = self.scrollbar.set

        # --- Переменные ---
        self.C = None
        self.D = None
        self.b = None
        self.running = False
        self.history = [] #будем хранить историю итераций

    def export_C(self):
        if self.C is None:
            messagebox.showerror("Ошибка", "Матрица C не сгенерирована или не загружена.")
            return
        filename = filedialog.asksaveasfilename(defaultextension=".csv", title="Сохранить матрицу C")
        if filename:
            self.export_matrix_to_csv(self.C, filename)

    def export_b(self):
        if self.b is None:
            messagebox.showerror("Ошибка", "Вектор b не сгенерирован или не загружен.")
            return
        filename = filedialog.asksaveasfilename(defaultextension=".csv", title="Сохранить вектор b")
        if filename:
            self.export_vector_to_csv(self.b, filename)

    def export_D(self):
        if self.D is None:
            messagebox.showerror("Ошибка", "Матрица D не сгенерирована.")
            return

        filename = filedialog.asksaveasfilename(defaultextension=".xlsx", title="Сохранить матрицы D")
        if filename:
            try:
                with pd.ExcelWriter(filename) as writer:
                    for i, matrix in enumerate(self.D):
                        df = pd.DataFrame(matrix)
                        df.to_excel(writer, sheet_name=f'D_{i}', index=False)
                messagebox.showinfo("Сохранено", f"Матрицы D успешно сохранены в файл {filename}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка сохранения файла: {e}")


    # --- Функции загрузки данных ---
    def load_matrix_from_csv(self, filename):
        """Загружает матрицу из CSV файла."""
        try:
            matrix = np.loadtxt(filename, delimiter=",")
            return matrix
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка загрузки файла: {e}")
            return None

    def load_vector_from_csv(self, filename):
        """Загружает вектор из CSV файла."""
        try:
            vector = np.loadtxt(filename, delimiter=",")
            return vector
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка загрузки файла: {e}")
            return None

    def export_matrix_to_csv(self, matrix, filename):
        """Сохраняет матрицу в CSV файл."""
        try:
            np.savetxt(filename, matrix, delimiter=",")
            messagebox.showinfo("Сохранено", "Матрица успешно сохранена.")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка сохранения файла: {e}")

    def export_vector_to_csv(self, vector, filename):
        """Сохраняет вектор в CSV файл."""
        try:
            np.savetxt(filename, vector, delimiter=",")
            messagebox.showinfo("Сохранено", "Вектор успешно сохранен.")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка сохранения файла: {e}")

    def export_C(self):
        if self.C is None:
            messagebox.showerror("Ошибка", "Матрица C не сгенерирована или не загружена.")
            return
        filename = filedialog.asksaveasfilename(defaultextension=".csv", title="Сохранить матрицу C")
        if filename:
            self.export_matrix_to_csv(self.C, filename)

    def export_b(self):
        if self.b is None:
            messagebox.showerror("Ошибка", "Вектор b не сгенерирован или не загружен.")
            return
        filename = filedialog.asksaveasfilename(defaultextension=".csv", title="Сохранить вектор b")
        if filename:
            self.export_vector_to_csv(self.b, filename)

    def load_C(self):
        filename = filedialog.askopenfilename(title="Выберите файл C")
        if filename:
            self.C = self.load_matrix_from_csv(filename)
            print("C загружена")

    def load_D(self):
         filename = filedialog.askopenfilename(title="Выберите файл D")
         if filename:
            self.D = self.load_matrix_from_csv(filename) #Тут надо доработать, т.к. D - это список матриц.
            print("D загружена")

    def load_b(self):
        filename = filedialog.askopenfilename(title="Выберите файл b")
        if filename:
            self.b = self.load_vector_from_csv(filename)
            print("b загружен")

    def generate_C(self):
        n = self.n_entry.get()
        lower_bound = self.lower_bound_C_entry.get()
        upper_bound = self.upper_bound_C_entry.get()
        if not n:
            messagebox.showerror("Ошибка", "Введите размер задачи n.")
            return
        self.C = self.generate_matrix(n, lower_bound, upper_bound)
        if self.C is not None:
            print("C сгенерирована")

    def generate_D(self):
        n = self.n_entry.get()
        k = self.k_entry.get()
        lower_bound = self.lower_bound_D_entry.get()
        upper_bound = self.upper_bound_D_entry.get()
        if not n or not k:
            messagebox.showerror("Ошибка", "Введите размер задачи n и число ограничений K.")
            return
        try:
            n = int(n)
            k = int(k)
            self.D = [self.generate_matrix(n, lower_bound, upper_bound) for _ in range(k)]  # Список матриц
            print("D сгенерирована")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при генерации D: {e}")

    def generate_b(self):
        k = self.k_entry.get()
        lower_bound = self.lower_bound_b_entry.get()
        upper_bound = self.upper_bound_b_entry.get()
        if not k:
            messagebox.showerror("Ошибка", "Введите число ограничений K.")
            return
        self.b = self.generate_vector(k, lower_bound, upper_bound)
        if self.b is not None:
            print("b сгенерирован")

    def generate_matrix(self, n, lower_bound, upper_bound):
        """Генерирует случайную матрицу размера n x n."""
        try:
            n = int(n)
            lower_bound = float(lower_bound)
            upper_bound = float(upper_bound)
            if lower_bound >= upper_bound:
                 raise ValueError("Нижняя граница должна быть меньше верхней")

            matrix = np.random.uniform(lower_bound, upper_bound, size=(n, n))
            return matrix
        except ValueError as e:
            messagebox.showerror("Ошибка", f"Некорректный размер матрицы или границы: {e}")
            return None

    def generate_vector(self, k, lower_bound, upper_bound):
        """Генерирует случайный вектор размера k."""
        try:
            k = int(k)
            lower_bound = float(lower_bound)
            upper_bound = float(upper_bound)
            if lower_bound >= upper_bound:
                 raise ValueError("Нижняя граница должна быть меньше верхней")
            vector = np.random.uniform(lower_bound, upper_bound, size=k)
            return vector
        except ValueError as e:
            messagebox.showerror("Ошибка", f"Некорректный размер вектора или границы: {e}")
            return None


    # --- Функции управления ---
    def start_calculation(self):
        try:
            n = int(self.n_entry.get())
            K = int(self.k_entry.get())
            N_max = int(self.n_max_entry.get())
            epsilon = float(self.epsilon_entry.get())
            step_rule = self.step_rule_combo.get()
            z_type = self.ztype_combo.get()

            # Проверка загрузки/генерации данных
            if self.C is None or self.D is None or self.b is None:
                messagebox.showerror("Ошибка", "Пожалуйста, загрузите или сгенерируйте матрицы C, D и вектор b.")
                return

            # Начальные lambda
            lambda0_str = self.lambda0_entry.get()
            if lambda0_str:
                try:
                    lambda0 = np.array([float(x) for x in lambda0_str.split(',')])
                    if len(lambda0) != K:
                        raise ValueError("Длина lambda0 не совпадает с K")
                except ValueError as e:
                    messagebox.showerror("Ошибка", f"Неверный формат lambda0: {e}")
                    return
            else:
                lambda0 = np.zeros(K) # Значение по умолчанию

            # Блокируем элементы управления
            self.start_button["state"] = "disabled"
            self.stop_button["state"] = "enabled"
            self.running = True
            self.log_text.delete("1.0", tk.END) # Очищаем лог

            # Запускаем вычисления в отдельном потоке (чтобы GUI не зависал)
            self.master.after(0, self.run_calculation, n, K, N_max, epsilon, step_rule, lambda0, z_type)

        except ValueError as e:
            messagebox.showerror("Ошибка", f"Ошибка ввода: {e}")

    def run_calculation(self, n, K, N_max, epsilon, step_rule, lambda0, z_type):
        """Запускает вычисления субградиентного метода."""
        self.X, self.lambda_k, self.history = alg.subgradient_method(self.C, self.D, self.b, lambda0, N_max, epsilon, step_rule, n, z_type)  # Запускаем алгоритм
        print("Тип self.history после вызова subgradient_method:", type(self.history))
        print("Содержимое self.history после вызова subgradient_method:", self.history)
        total_time = self.history['time common']
        self.total_time_value.set(f"{total_time:.2f}")
        self.update_gui()
        self.stop_calculation()  # Разблокируем кнопки


    def stop_calculation(self):
        self.start_button["state"] = "enabled"
        self.stop_button["state"] = "disabled"
        self.running = False

    def save_log(self):
        filename = filedialog.asksaveasfilename(defaultextension=".csv",
                                                 filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*")])
        if filename:
            try:
                with open(filename, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    # Записываем заголовки
                    writer.writerow(["Iteration", "Lambda", "Subgradient", "Objective Value", "Violation", "Time"])
                    # Записываем данные
                    for item in self.history:
                        writer.writerow([item["iteration"], item["lambda"], item["subgradient"], item["objective_value"], item["violation"], item["time"]])
                messagebox.showinfo("Сохранено", "Лог успешно сохранен.")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка сохранения: {e}")


    # --- Функции обновления GUI ---
    def update_gui(self):
        """Обновляет GUI после завершения вычислений."""
        if not self.history:
            messagebox.showinfo("Информация", "Вычислений не было произведено.")
            return

        # Обновляем таблицу X (отображаем последнюю матрицу)
        self.update_table(self.X)  # Передаем self.X

        # Обновляем график
        self.update_plot()

        # Обновляем журнал
        self.update_log()

    def update_table(self, X):  # Принимаем C как аргумент
        """Обновляет таблицу Treeview матрицей X, заменяя 1 значениями из C."""
        X = np.array(X)  # Преобразуйте X в массив NumPy (если это еще не сделано)
        C = np.array(self.C)  # Преобразуйте C в массив NumPy

        # Очистить таблицу перед заполнением
        for item in self.table.get_children():
            self.table.delete(item)

        # Определить структуру таблицы (количество столбцов)
        num_cols = X.shape[1]
        column_ids = [str(i + 1) for i in range(num_cols)]  # ID столбцов
        self.table["columns"] = column_ids
        self.table["show"] = "headings"  # Скрыть пустой столбец

        # Задать заголовки столбцов
        for col_id in column_ids:
            self.table.heading(col_id, text=col_id, command=lambda: None)
            self.table.column(col_id, width=30,  minwidth=30)  # УМЕНЬШЕННАЯ ШИРИНА СТОЛБЦА
            self.table.column(col_id, anchor='center')  # Центрирование текста

        # Заполнить таблицу данными
        for row in range(X.shape[0]):
            formatted_row = []
            for col in range(X.shape[1]):
                if X[row, col] > 0.5:  # Если значение в X больше 0.5 (считаем это 1)
                    formatted_row.append(f"{self.X[row, col]:.2f}")  # Берем значение из C и форматируем
                else:
                    formatted_row.append("0")  # Если 0, то выводим 0.00
            self.table.insert("", "end", values=formatted_row)

        # Обновляем scrollregion
        #self.table.pack(expand=False, fill="both", padx=10, pady=10)

    def update_plot(self):
        """Обновляет график зависимости нарушения от номера итерации."""
        if isinstance(self.history, dict) and "violations" in self.history:
            violations = self.history["violations"]
            iterations = list(range(len(violations)))  # Создаем список итераций

            print("Нарушения:", violations)
            print("Итерации:", iterations)

            self.ax.clear()
            self.ax.plot(iterations, violations)
            self.ax.set_xlabel("Итерация")
            self.ax.set_ylabel("Нарушение (L1-норма)")
            self.ax.set_title("Зависимость нарушения от итерации")
            self.ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            self.fig.canvas.draw()
        else:
            print("Ошибка: self.history имеет неверную структуру.")

    def update_log(self):
        """Обновляет журнал."""
        self.log_text.delete("1.0", tk.END)  # Очищаем журнал

        if not self.history:
            self.log_text.insert(tk.END, "Вычислений не было произведено.\n")
            return

        for i, (lambda_val, grad, cost, violation, time_iter) in enumerate(
                zip(self.history['lambda'], self.history['grad'], self.history['cost'],
                    self.history['violations'], self.history['time iter'])):
            self.log_text.insert(tk.END, f"Итерация {i + 1}:\n")
            self.log_text.insert(tk.END, f"  lambda: {lambda_val}\n")
            self.log_text.insert(tk.END, f"  grad: {grad}\n")
            self.log_text.insert(tk.END, f"  cost: {cost:.2f}\n")
            self.log_text.insert(tk.END, f"  нарушение: {violation:.2f}\n")
            self.log_text.insert(tk.END, f"  Время итерации: {time_iter:.4f} секунд\n")
            self.log_text.insert(tk.END, "\n")

        # Добавляем общее время в конец журнала
        total_time = self.history['time common']
        self.log_text.insert(tk.END, f"Общее время работы: {total_time:.2f} секунд.\n")
        self.log_text.see(tk.END)


# --- Запуск GUI ---
if __name__ == "__main__":
    root = tk.Tk()
    ttk.Style().theme_use("clam")
    ttk.Style().configure(".", font = ('Helvetica', 12, 'bold'))

    gui = SubgradientGUI(root)
    root.mainloop()
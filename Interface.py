import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import time
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import Algorithm as alg

# --- Функции для субградиентного метода (ЗАГЛУШКИ - нужно реализовать!) ---

def subgradient_method(C, D, b, lambda0, N_max, epsilon, step_rule, n):
    """
    Реализация субградиентного метода.

    Args:
        C: Матрица цен (n x n).
        D: Список матриц ограничений (K x n x n).
        b: Вектор ограничений (K).
        lambda0: Начальные значения множителей Лагранжа (K).
        N_max: Максимальное число итераций.
        epsilon: Порог точности.
        step_rule: Правило выбора шага ("1/n" или "const").

    Returns:
        history: История итераций (список словарей).
    """

    history = []  # Список для хранения информации о каждой итерации.
    lambda_k = lambda0.copy()  # Текущее значение lambda
    X = np.zeros((n,n)) #начальное приближение решения
    for i in range(N_max):
        start_time = time.time()

        # 1. Рассчитать X (матрицу решений) - нужно реализовать!
        #    На основе C, D и lambda_k
        X = calculate_X(C, D, lambda_k) #заглушка

        # 2. Рассчитать субградиент двойственной функции - нужно реализовать!
        subgradient = calculate_subgradient(D, X, b) #заглушка

        # 3. Рассчитать целевую функцию - нужно реализовать!
        objective_value = calculate_objective(C,X,D,b,lambda_k) #заглушка

        # 4. Рассчитать величину нарушения (L1-норма субградиента)
        violation = np.sum(np.abs(subgradient))

        # 5. Выбрать шаг
        if step_rule == "1/n":
            step = 1 / (i + 1)  # Пример правила шага
        elif step_rule == "const":
            step = 0.01        # Пример постоянного шага
        else:
            raise ValueError("Неверное правило шага")

        # 6. Обновить lambda
        lambda_k = lambda_k - step * subgradient

        iteration_time = time.time() - start_time

        history.append({
            "iteration": i,
            "X": X.copy(),  # Копируем X, чтобы не менялся при следующих итерациях
            "lambda": lambda_k.copy(),
            "subgradient": subgradient.copy(),
            "objective_value": objective_value,
            "violation": violation,
            "time": iteration_time
        })

        if violation < epsilon:
            print(f"Достигнута точность на итерации {i}")
            break

    return history


def calculate_X(C, D, lambda_k):
  #Заглушка - надо реализовать расчет X
  n = C.shape[0]
  return np.random.rand(n,n)

def calculate_subgradient(D, X, b):
  #Заглушка - надо реализовать расчет субградиента
  K = len(b)
  return np.random.rand(K)

def calculate_objective(C,X,D,b,lambda_k):
  #Заглушка - надо реализовать расчет целевой функции
  return np.random.rand(1)[0]


# --- Основной класс GUI ---
class SubgradientGUI:
    def __init__(self, master):
        self.master = master
        master.title("Анализ Субградиентного Метода")

        # --- Параметры задачи ---
        self.frame_params = ttk.LabelFrame(master, text="Параметры Задачи")
        self.frame_params.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        ttk.Label(self.frame_params, text="Размер задачи (n):").grid(row=0, column=0, sticky="w")
        self.n_entry = ttk.Entry(self.frame_params)
        self.n_entry.grid(row=0, column=1, sticky="w")

        ttk.Label(self.frame_params, text="Число ограничений (K):").grid(row=1, column=0, sticky="w")
        self.k_entry = ttk.Entry(self.frame_params)
        self.k_entry.grid(row=1, column=1, sticky="w")

        # Кнопки для загрузки данных И автозаполнения
        ttk.Button(self.frame_params, text="Загрузить C", command=self.load_C).grid(row=2, column=0, pady=5)
        ttk.Button(self.frame_params, text="Сгенерировать C", command=self.generate_C).grid(row=2, column=1, pady=5)

        ttk.Button(self.frame_params, text="Загрузить D", command=self.load_D).grid(row=3, column=0, pady=5)
        ttk.Button(self.frame_params, text="Сгенерировать D", command=self.generate_D).grid(row=3, column=1, pady=5)

        ttk.Button(self.frame_params, text="Загрузить b", command=self.load_b).grid(row=4, column=0, pady=5)
        ttk.Button(self.frame_params, text="Сгенерировать b", command=self.generate_b).grid(row=4, column=1, pady=5)


        # --- Параметры алгоритма ---
        self.frame_algo = ttk.LabelFrame(master, text="Параметры Алгоритма")
        self.frame_algo.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        ttk.Label(self.frame_algo, text="Макс. итераций (N_max):").grid(row=0, column=0, sticky="w")
        self.n_max_entry = ttk.Entry(self.frame_algo)
        self.n_max_entry.grid(row=0, column=1, sticky="w")

        ttk.Label(self.frame_algo, text="Порог точности (ε):").grid(row=1, column=0, sticky="w")
        self.epsilon_entry = ttk.Entry(self.frame_algo)
        self.epsilon_entry.grid(row=1, column=1, sticky="w")

        ttk.Label(self.frame_algo, text="Правило шага:").grid(row=2, column=0, sticky="w")
        self.step_rule_combo = ttk.Combobox(self.frame_algo, values=["1/n", "const"])
        self.step_rule_combo.grid(row=2, column=1, sticky="w")
        self.step_rule_combo.set("1/n")  # Значение по умолчанию

        ttk.Label(self.frame_algo, text="Начальные λ₀ (опционально):").grid(row=3, column=0, sticky="w")
        self.lambda0_entry = ttk.Entry(self.frame_algo)
        self.lambda0_entry.grid(row=3, column=1, sticky="w")

        # --- Кнопки управления ---
        self.frame_controls = ttk.Frame(master)
        self.frame_controls.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

        self.start_button = ttk.Button(self.frame_controls, text="Запуск", command=self.start_calculation)
        self.start_button.grid(row=0, column=0, padx=5)

        self.stop_button = ttk.Button(self.frame_controls, text="Остановка", command=self.stop_calculation, state="disabled")
        self.stop_button.grid(row=0, column=1, padx=5)

        self.save_log_button = ttk.Button(self.frame_controls, text="Сохранить лог", command=self.save_log)
        self.save_log_button.grid(row=0, column=2, padx=5)

        # --- Вывод результатов ---
        self.notebook = ttk.Notebook(master)
        self.notebook.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

        # Таблица X
        self.table_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.table_frame, text="Матрица X")

        # Создаем Canvas для размещения Treeview
        self.table_canvas = tk.Canvas(self.table_frame)
        self.table_canvas.pack(side="left", fill="both", expand=True)

        # Добавляем Scrollbars
        self.table_y_scrollbar = ttk.Scrollbar(self.table_frame, orient="vertical", command=self.table_canvas.yview)
        self.table_y_scrollbar.pack(side="right", fill="y")
        self.table_x_scrollbar = ttk.Scrollbar(self.table_frame, orient="horizontal", command=self.table_canvas.xview)
        self.table_x_scrollbar.pack(side="bottom", fill="x")

        # Настраиваем Canvas
        self.table_canvas.configure(yscrollcommand=self.table_y_scrollbar.set,
                                    xscrollcommand=self.table_x_scrollbar.set)
        self.table_canvas.bind("<Configure>",
                               lambda e: self.table_canvas.configure(scrollregion=self.table_canvas.bbox("all")))

        # Создаем внутренний Frame для Treeview
        self.table_inner_frame = ttk.Frame(self.table_canvas)
        self.table_canvas.create_window((0, 0), window=self.table_inner_frame, anchor="nw")

        # Создаем Treeview
        self.table = ttk.Treeview(self.table_inner_frame)
        self.table.pack(side="left", fill="both", expand=True)

        # График нарушения
        self.plot_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.plot_frame, text="График Нарушения")
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Журнал
        self.log_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.log_frame, text="Журнал")
        self.log_text = tk.Text(self.log_frame, wrap="word")
        self.log_text.pack(fill="both", expand=True)
        self.scrollbar = ttk.Scrollbar(self.log_frame, command=self.log_text.yview)
        self.scrollbar.pack(side="right", fill="y")
        self.log_text['yscrollcommand'] = self.scrollbar.set

        # --- Переменные ---
        self.C = None
        self.D = None
        self.b = None
        self.running = False
        self.history = [] #будем хранить историю итераций


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

    # --- Функции автозаполнения ---
    def generate_matrix(self, n):
        """Генерирует случайную матрицу размера n x n."""
        try:
            n = int(n)
            matrix = np.random.rand(n, n)
            return matrix
        except ValueError:
            messagebox.showerror("Ошибка", "Некорректный размер матрицы.")
            return None

    def generate_vector(self, k):
        """Генерирует случайный вектор размера k."""
        try:
            k = int(k)
            vector = np.random.rand(k)
            return vector
        except ValueError:
            messagebox.showerror("Ошибка", "Некорректный размер вектора.")
            return None


    def generate_C(self):
        n = self.n_entry.get()
        if not n:
            messagebox.showerror("Ошибка", "Введите размер задачи n.")
            return
        self.C = self.generate_matrix(n)
        if self.C is not None:
            print("C сгенерирована")

    def generate_D(self):
        n = self.n_entry.get()
        k = self.k_entry.get()
        if not n or not k:
            messagebox.showerror("Ошибка", "Введите размер задачи n и число ограничений K.")
            return
        try:
            n = int(n)
            k = int(k)
            self.D = [self.generate_matrix(n) for _ in range(k)]  # Список матриц
            print("D сгенерирована")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при генерации D: {e}")

    def generate_b(self):
        k = self.k_entry.get()
        if not k:
            messagebox.showerror("Ошибка", "Введите число ограничений K.")
            return
        self.b = self.generate_vector(k)
        if self.b is not None:
            print("b сгенерирован")


    # --- Функции управления ---
    def start_calculation(self):
        try:
            n = int(self.n_entry.get())
            K = int(self.k_entry.get())
            N_max = int(self.n_max_entry.get())
            epsilon = float(self.epsilon_entry.get())
            step_rule = self.step_rule_combo.get()

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
            self.master.after(0, self.run_calculation, n, K, N_max, epsilon, step_rule, lambda0)

        except ValueError as e:
            messagebox.showerror("Ошибка", f"Ошибка ввода: {e}")

    def run_calculation(self, n, K, N_max, epsilon, step_rule, lambda0):
        """Запускает вычисления субградиентного метода."""
        self.X, self.lambda_k, self.history = alg.subgradient_method(self.C, self.D, self.b, lambda0, N_max, epsilon, step_rule, n)  # Запускаем алгоритм
        print("Тип self.history после вызова subgradient_method:", type(self.history))
        print("Содержимое self.history после вызова subgradient_method:", self.history)
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

    def update_table(self, X):
        """Обновляет таблицу Treeview матрицей X."""
        X = np.array(X)  # Преобразуйте X в массив NumPy (если это еще не сделано)

        # Очистить таблицу перед заполнением
        for item in self.table.get_children():
            self.table.delete(item)

        # Определить структуру таблицы (количество столбцов)
        num_cols = X.shape[1]
        column_ids = [str(i) for i in range(num_cols)]  # ID столбцов
        self.table["columns"] = column_ids
        self.table["show"] = "headings"  # Скрыть пустой столбец

        # Задать заголовки столбцов
        for col_id in column_ids:
            self.table.heading(col_id, text=col_id)
            self.table.column(col_id, width=30)  # УМЕНЬШЕННАЯ ШИРИНА СТОЛБЦА
            self.table.column(col_id, anchor='center')  # Центрирование текста

        # Заполнить таблицу данными
        for row in range(X.shape[0]):
            formatted_row = [f"{x:.0f}" for x in X[row, :]]  # Форматирование чисел
            self.table.insert("", "end", values=formatted_row)

        # Обновляем scrollregion
        self.table_canvas.configure(scrollregion=self.table_canvas.bbox("all"))

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
        """Обновляет текстовый журнал информацией об итерациях."""
        if isinstance(self.history, dict):
            num_iterations = len(self.history['lambda'])
            for i in range(num_iterations):
                log_entry = (f"Итерация: {i}\n"
                             f"  λ: {self.history['lambda'][i]}\n"
                             f"  Субградиент: {self.history['grad'][i]}\n"
                             f"  Целевая функция: {self.history['cost'][i]}\n"
                             f"  Нарушение: {self.history['violations'][i]}\n"
                             f"  Время: {self.history['time'][i]:.4f} сек\n\n")
                self.log_text.insert(tk.END, log_entry)

            self.log_text.see(tk.END)  # Прокручиваем в конец


# --- Запуск GUI ---
if __name__ == "__main__":
    root = tk.Tk()
    gui = SubgradientGUI(root)
    root.mainloop()
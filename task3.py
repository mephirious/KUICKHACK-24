import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from openai import OpenAI
import locale
import matplotlib.dates as mdates
from matplotlib.font_manager import FontProperties
from matplotlib.colors import LinearSegmentedColormap

locale.setlocale(locale.LC_ALL, 'ru_RU.UTF-8')

def load_data(uploaded_file):
    try:
        data = pd.read_csv(uploaded_file)
        data['Дата'] = pd.to_datetime(data['Дата'], errors='coerce')
        data.set_index('Дата', inplace=True)
        data['Сумма'] = pd.to_numeric(data['Сумма'], errors='coerce')
        return data
    except Exception as e:
        st.error(f"Error processing the uploaded file: {e}")
        return None



def adjust_cmap(cmap_name, min_val=0.2, max_val=1.0):
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(min_val, max_val, 256))
    new_cmap = LinearSegmentedColormap.from_list("trimmed_" + cmap_name, colors)
    return new_cmap

def load_park_data(filepath):
    try:
        data = pd.read_csv(filepath)
        data.set_index('Название', inplace=True)
        return data
    except Exception as e:
        st.error(f"Не удалось загрузить данные из файла: {e}")
        return pd.DataFrame()

def load_canteen_data(filepath):
    try:
        data = pd.read_csv(filepath)
        data.set_index('Название', inplace=True)
        return data
    except Exception as e:
        st.error(f"Не удалось загрузить данные из файла: {e}")
        return pd.DataFrame()

def plot_financial_data(data):
    income_data = data[data['Доходы/Расходы'] == 'Доходы']['Сумма'].resample('D').sum()
    expense_data = data[data['Доходы/Расходы'] == 'Расходы']['Сумма'].resample('D').sum()

    fig, ax = plt.subplots(figsize=(15, 6))

    income_data.plot(kind='bar', ax=ax, color='#213a85', label='Доходы', width=0.4, position=0)
    expense_data.plot(kind='bar', ax=ax, color='#e43b29', label='Расходы', width=0.4, position=1)

    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d'))

    plt.title('Динамика доходов и расходов по дням', fontname='Segoe UI', fontsize=16, weight='bold')
    plt.xlabel('Дата', fontname='Segoe UI', fontsize=14)
    plt.ylabel('Сумма', fontname='Segoe UI', fontsize=14)
    plt.xticks(rotation=0)
    plt.legend()
    plt.tight_layout()

    st.pyplot(fig)

def get_cmap_colors(cmap_name, num_colors):
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(0, 1, num_colors))
    print("Returned colors:", colors)
    return colors

def plot_financial_pie_charts(data):
    income_data = data[data['Доходы/Расходы'] == 'Доходы']
    expense_data = data[data['Доходы/Расходы'] == 'Расходы']

    def group_small_categories(series, threshold=0.05):
        small_categories = series[series / series.sum() < threshold]
        large_categories = series[series / series.sum() >= threshold]
        if not small_categories.empty:
            other_sum = small_categories.sum()
            large_categories = pd.concat([large_categories, pd.Series([other_sum], index=['Мелкие'])])
        return large_categories

    income_by_category = group_small_categories(income_data.groupby('Категория')['Сумма'].sum())
    expense_by_category = group_small_categories(expense_data.groupby('Категория')['Сумма'].sum())

    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    font = FontProperties(family='Segoe UI', size=16)

    wedges1, texts1, autotexts1 = axs[0].pie(income_by_category, labels=income_by_category.index,
                                             autopct='%1.1f%%',
                                             startangle=140, radius=1.2,
                                             colors=get_cmap_colors(adjust_cmap('Blues', 0.2, 0.7), len(income_by_category)),
                                             textprops={'fontsize': 16, 'fontfamily': 'Segoe UI'})
    axs[0].set_title('Доходы по категориям', pad=70, fontdict={'fontweight': 'bold', 'family': 'Segoe UI', 'size': 16})

    wedges2, texts2, autotexts2 = axs[1].pie(expense_by_category, labels=expense_by_category.index,
                                             autopct='%1.1f%%',
                                             startangle=140, radius=1.2,
                                             colors=get_cmap_colors(adjust_cmap('Reds', 0.2, 0.7), len(income_by_category)),
                                             textprops={'fontsize': 16, 'fontfamily': 'Segoe UI'})
    axs[1].set_title('Расходы по категориям', pad=70, fontdict={'fontweight': 'bold', 'family': 'Segoe UI', 'size': 16})

    axs[0].legend(wedges1, income_by_category.index, prop=font, title="Категории", loc="center left", bbox_to_anchor=(1.1, 1))
    axs[1].legend(wedges2, expense_by_category.index, prop=font, title="Категории", loc="center right", bbox_to_anchor=(-0.1,1))

    plt.tight_layout()
    st.pyplot(fig)

def prepare_prompt(data):
    csv_string = data.to_string()
    return csv_string

def send_to_gpt_ai(prompt, section, country, social_status):
    api_key = 'GOOGLE-API-KEY'
    if not api_key:
        return "API key not set. Please configure the environment variable."

    sch = [
        f'''ИНСТРУКЦИЯ: Используй следующие пошаговые инструкции, чтобы ответить на действия пользователя.

        Учитывай страну {country} и социальный статус {social_status}
        
        Шаг 1. Пользователь предоставит выписку доходов и расходов с датой, категорией, описанием и суммой. Ты должен рассортировать по категориям затраты

        Шаг 2. Подсчитай общую сумму расходов каждой категории и прочитай ее описание

        Шаг 3. Подсчитай общее количество доходов и прочитай источник дохода

        Шаг 4. Ты должен вывести все категории и написать общую сумму каждой категории и ее описание

        ПРИМЕР ОТВЕТА: 

        В мае 2024 года казахстанский студент совершил следующие траты:
        
        1. Транспорт: 7200.0 KZT на проезд на общественном транспорте и такси.
        
        2. Еда: 27,400.0 KZT на завтраки, обеды, ужины и покупку продуктов.
        
        3. Досуг: 27,500.0 KZT на посещение кинотеатра, музея, концерта, парка аттракционов, боулинга, выставок и кино.
        
        4. Учеба: 9100.0 KZT на покупку учебников.
        
        5. Доходы: 73,000.0 KZT доходы от подработки, стипендии, фриланс-проектов и участия в маркетинговом исследовании.
        
        Эти расходы отражают жизненный стиль и интересы студента в мае 2024 года.
        
        ЗАПРОС: {prompt}
        ''',
                f'''ИНСТРУКЦИЯ: Используй следующие пошаговые инструкции, чтобы ответить на действия пользователя.
                
        Шаг 0. Учитывай страну {country} и социальный статус {social_status}
        
        Шаг 1. Пользователь предоставит выписку доходов и расходов с датой, категорией, описанием и суммой. Ты посчитать сколько можно сэкономить
        
        Шаг 2. Раздели расходы по категориям
        
        Шаг 3. Напиши для каждой категории рекомендации
        Шаг 4. Напиши на сколько можно сократить расходы
        
        Шаг 5. Рассчитай сколько можно сэкономить в месяц
        
        ПРИМЕР ОТВЕТА: 
        
        1. Питание: Вы тратите слишком много на еду. Попробуйте готовить дома чаще, а не посещать столовые и рестораны каждый день. Сократите количество обедов и ужинов вне дома с 20 до 10 (50% сокращение). Так вы сэкономите 8100 тенге в месяц (1500 * 10 * 2).
        
        2. Транспорт: Снизьте расходы на общественный транспорт, используя его более эффективно. Попробуйте пользоваться такси меньше и передвигаться пешком или на велосипеде. Сократите количество поездок с 60 до 30 (50% сокращение). Также можно использовать студенческие скидки на проезд. В среднем в Казахстане можно уложиться в 5000 тенге в месяц на транспорт.
        
        3. Учеба: Покупка учебников также занимает большую часть вашего бюджета. Попробуйте искать вторичные и более доступные источники для материалов. Планируйте покупки и сократите расходы на учебники с 9600 тенге до 2000 тенге в месяц (79% сокращение).
        
        4. Досуг: Посещение развлекательных мероприятий также может быть дорогим удовольствием. Подумайте о более бюджетных развлечениях или планируйте их реже. Уменьшите расходы на досуг с 4100 тенге до 2000 тенге в месяц (51% сокращение).
        
        5. Общие расходы: В среднем в Казахстане можно сэкономить до 22000 тенге в месяц, следуя вышеперечисленным рекомендациям и сокращая лишние траты.
        
        Сосредоточьте на этих областях, чтобы улучшить управление своим бюджетом и сэкономить
        
        ЗАПРОС: {prompt}
        '''
            ]


    content = sch[section]

    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "assistant",
                 "content": "Ты финансовый ассистент для грамотного планирования бюджета"},
                {"role": "user", "content": content}
            ]
        )
        if response:
            return response.choices[0].message.content
        else:
            return "Ошибка"
    except Exception as e:
        return f"An error occurred: {e}"


def main():
    st.title("Финансовый помощник")

    user_name = st.text_input("Введите ваше имя:", "")
    social_status = st.selectbox("Выберите ваш социальный статус:", ["Студент", "Работающий", "Пенсионер"])
    country = st.selectbox("В какой стране вы находитесь?:", ["Казахстан", "Россия", "Англия"])
    city = st.selectbox("Выберите ваш город:", ["Астана", "Москва", "Лондон"])

    uploaded_file = st.file_uploader("Загрузите свой CSV файл", type="csv")
    if uploaded_file is not None:
        if city == "Астана" and social_status == "Студент":
            park_data = load_park_data("par.csv")
            load_canteen = load_canteen_data("canteen.csv")
        data = load_data(uploaded_file)
        st.write("Предварительный обзор ваших данных:", fontweight='bold', pad='50')
        st.dataframe(data, width=700, height=200)

        if data is not None:
            prompt = prepare_prompt(data)

            if st.button("Проанализировать"):
                intro_analysis = send_to_gpt_ai(prompt, 0, country, social_status)
                st.subheader('Введение и обзор данных')
                plot_financial_data(data)
                st.write(intro_analysis)

                pie_chart_analysis = send_to_gpt_ai(prompt, 1, country, social_status)
                st.subheader('Рекомендации по категориям доходов и расходов')
                plot_financial_pie_charts(data)
                st.write(pie_chart_analysis)

                if city == "Астана" and social_status == "Студент":
                    if park_data is not None:
                        st.write("")
                        st.write('Список мест для бесплатного досуга:')
                        st.dataframe(park_data, width=700, height=200)
                        st.write("")
                        st.write('Список столовых с указанием среднего чека:')
                        st.dataframe(load_canteen, width=700, height=200)

if __name__ == "__main__":
    main()

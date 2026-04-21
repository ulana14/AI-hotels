import streamlit as st
from gigachat import GigaChat
import os
import json
import glob
import re
import asyncio
import time
import pandas as pd
from collections import Counter

def fix_async_loop():
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

os.environ['no_proxy'] = 'sberbank.ru,openstreetmap.org,overpass-api.de'

# ==================== ЗАГРУЗКА ДАННЫХ КЛАССИФИКАЦИИ ====================
@st.cache_data
def load_reviews_data():
    """Загрузка данных с результатами классификации отзывов"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "hotel_reviews_extended.csv")
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return df
    else:
        # Если файла нет, пробуем загрузить оригинальный
        csv_path = os.path.join(current_dir, "all_cities_raw.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Добавляем бинарную классификацию
            df['label'] = (df['user_rating'] >= 8).astype(int)
            return df
        return None

@st.cache_data
def load_hotel_database():
    combined = []
    file_names = []
    current_dir = os.path.dirname(os.path.abspath(__file__))
    files = glob.glob(os.path.join(current_dir, "*.json"))
    
    for f_path in files:
        try:
            with open(f_path, 'r', encoding='utf-8-sig') as f:
                data = json.load(f)
                if isinstance(data, list):
                    combined.extend(data)
                else:
                    combined.append(data)
                file_names.append(os.path.basename(f_path))
        except:
            continue
    return combined, file_names

class HotelAIAssistant:
    def __init__(self, creds):
        self.creds = creds
        self.all_hotels_data, self.file_names = load_hotel_database()
        self.reviews_df = load_reviews_data()
        
        # Словарь с именами файлов и достопримечательностями
        self.city_map = {
            "Москва": {"file": "moscow", "landmarks": "Кремль, Красная площадь, парк Зарядье, Большой театр"},
            "Санкт-Петербург": {"file": "saint_petersburg", "landmarks": "Эрмитаж, Исаакиевский собор, Невский проспект, Дворцовая площадь"},
            "Казань": {"file": "kazan", "landmarks": "Казанский Кремль, мечеть Кул-Шариф, набережная озера Кабан, улица Баумана"},
            "Иркутск": {"file": "irkutsk", "landmarks": "озеро Байкал, Иркутская слобода, набережная Ангары"},
            "Красноярск": {"file": "krasnoyarsk", "landmarks": "заповедник Столбы, набережная Енисея, остров Татышев"},
            "Нижний Новгород": {"file": "nizhniy_novgorod", "landmarks": "Нижегородский Кремль, Чкаловская лестница, Стрелка"},
            "Новосибирск": {"file": "novosibirsk", "landmarks": "Оперный театр, Новосибирский зоопарк, Академгородок"},
            "Пермь": {"file": "perm", "landmarks": "Пермская художественная галерея, набережная Камы, памятник Пермяк-соленые уши"},
            "Петропавловск-Камчатский": {"file": "petropavlovsk_kamchatsky", "landmarks": "Халактырский пляж, Авачинская бухта, вулканы Камчатки"},
            "Владимир": {"file": "vladimir", "landmarks": "Золотые ворота, Успенский собор, Дмитриевский собор"},
            "Владивосток": {"file": "vladivostok", "landmarks": "Золотой мост, остров Русский, маяк Эгершельд"},
            "Волгоград": {"file": "volgograd", "landmarks": "Мамаев курган, музей-панорама Сталинградская битва"},
            "Воронеж": {"file": "voronezh", "landmarks": "Адмиралтейская площадь, памятник Котенку с улицы Лизюкова, Корабль Гото Предестинация"},
            "Сочи": {"file": "sochi", "landmarks": "Олимпийский парк, Красная Поляна, парк Ривьера, Морской порт"}
        }

    def get_reviews_analysis(self, city_name):
        """Анализ отзывов для конкретного города"""
        if self.reviews_df is None:
            return None, "Данные отзывов не загружены"
        
        city_df = self.reviews_df[self.reviews_df['city'] == city_name]
        
        if len(city_df) == 0:
            return None, f"Отзывы для города '{city_name}' не найдены"
        
        # Статистика
        total_reviews = len(city_df)
        avg_rating = city_df['user_rating'].mean()
        
        # Бинарная классификация (если есть колонка label)
        if 'label' in city_df.columns:
            positive_count = (city_df['label'] == 1).sum()
            negative_count = (city_df['label'] == 0).sum()
        else:
            positive_count = (city_df['user_rating'] >= 8).sum()
            negative_count = (city_df['user_rating'] < 8).sum()
        
        positive_ratio = positive_count / total_reviews if total_reviews > 0 else 0
        
        # Ключевые слова из позитивных отзывов
        positive_reviews = city_df[city_df['user_rating'] >= 8]['text'].tolist()
        negative_reviews = city_df[city_df['user_rating'] <= 6]['text'].tolist()
        
        # Простой анализ частых слов (без стоп-слов)
        stop_words = {'в', 'на', 'и', 'с', 'по', 'к', 'у', 'о', 'от', 'за', 'из', 'для', 
                      'не', 'что', 'как', 'это', 'а', 'но', 'или', 'мы', 'вы', 'ты', 'я',
                      'был', 'была', 'были', 'есть', 'очень', 'все', 'весь', 'еще', 'уже',
                      'только', 'также', 'который', 'свой', 'себя', 'там', 'тут', 'здесь'}
        
        pos_words = []
        for text in positive_reviews[:20]:  # Ограничиваем для скорости
            words = re.findall(r'[а-яё]{4,}', str(text).lower())
            pos_words.extend([w for w in words if w not in stop_words])
        
        neg_words = []
        for text in negative_reviews[:20]:
            words = re.findall(r'[а-яё]{4,}', str(text).lower())
            neg_words.extend([w for w in words if w not in stop_words])
        
        pos_counter = Counter(pos_words).most_common(10)
        neg_counter = Counter(neg_words).most_common(10)
        
        # Отели с лучшими и худшими отзывами
        hotel_stats = city_df.groupby('hotel').agg({
            'user_rating': ['mean', 'count']
        }).round(2)
        hotel_stats.columns = ['avg_rating', 'review_count']
        hotel_stats = hotel_stats[hotel_stats['review_count'] >= 2].sort_values('avg_rating', ascending=False)
        
        best_hotels = hotel_stats.head(5).index.tolist()
        worst_hotels = hotel_stats.tail(5).index.tolist()
        
        analysis = {
            "total_reviews": total_reviews,
            "avg_rating": round(avg_rating, 2),
            "positive_count": positive_count,
            "negative_count": negative_count,
            "positive_ratio": round(positive_ratio * 100, 1),
            "top_positive_words": [w[0] for w in pos_counter],
            "top_negative_words": [w[0] for w in neg_counter],
            "best_hotels": best_hotels,
            "worst_hotels": worst_hotels,
            "sample_positive": positive_reviews[:3] if positive_reviews else [],
            "sample_negative": negative_reviews[:3] if negative_reviews else []
        }
        
        return analysis, ""

    def get_market_context(self, city_display_name):
        city_info = self.city_map.get(city_display_name, {"file": city_display_name.lower(), "landmarks": ""})
        search_name = city_info["file"]
        
        competitors = []
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        target_file = None
        for f_path in glob.glob(os.path.join(current_dir, "*.json")):
            f_name = os.path.basename(f_path).lower()
            if search_name in f_name:
                target_file = f_path
                break
        
        if target_file:
            try:
                with open(target_file, 'r', encoding='utf-8-sig') as f:
                    data = json.load(f)
                    competitors = data if isinstance(data, list) else [data]
            except Exception as e:
                return {}, f"Ошибка в файле {target_file}: {e}"
        
        if not competitors:
            return {}, f"Данные для города '{city_display_name}' не найдены."

        trash_words = [
            'скрыть', 'показать', 'информация', 'услуги', 'оплачивается', 
            'дополнительно', 'подробнее', 'цены', 'руб', 'бесплатно',
            'безопасность', 'удобства в номерах', 'язык персонала', 
            'правила проживания', 'сервисы', 'общие', 'другое', 
            'описание услуг', 'интернет', 'питание и напитки',
            'медиа и технологии', 'на свежем воздухе', 'для бизнеса', 'детям', 'вне помещения',
            'в номере', 'в ванной комнате', 'огнетушители', 'медиа и развлечения', 
            'датчики дыма', 'набор полотенец', 'тапочки', 'транспорт', 'русский', 'английский', 
            'антибактериальный гель', 'парковка', 'бесплатный Wi-Fi'
        ] 
        all_services = []
        for h in competitors:
            services = h.get('services', [])
            if isinstance(services, list):
                for s in services:
                    s_clean = s.strip()
                    if (len(s_clean) > 3 and not re.search(r'\d', s_clean) and 
                        not any(trash in s_clean.lower() for trash in trash_words)):
                        s_clean = re.sub(r'([а-я])([А-Я])', r'\1 \2', s_clean)
                        all_services.append(s_clean)

        service_counts = Counter(all_services).most_common(30)
        
        # Получаем анализ отзывов
        reviews_analysis, _ = self.get_reviews_analysis(city_display_name)
        
        context_data = {
            "count": len(competitors),
            "top_services": service_counts,
            "names": [h.get('name', 'Без названия') for h in competitors],
            "raw_services_text": ", ".join([s[0] for s in service_counts]),
            "landmarks": city_info["landmarks"],
            "reviews_analysis": reviews_analysis  # ДОБАВЛЯЕМ АНАЛИЗ ОТЗЫВОВ
        }
        return context_data, ""

    def generate_description(self, hotel_name, city_name, features, market_data):
        fix_async_loop()
        
        # Формируем часть промпта с анализом отзывов
        reviews_info = ""
        if market_data.get('reviews_analysis'):
            ra = market_data['reviews_analysis']
            reviews_info = (
                f"\n\n📊 АНАЛИЗ ОТЗЫВОВ ПО ГОРОДУ:"
                f"\n- Всего проанализировано отзывов: {ra['total_reviews']}"
                f"\n- Средний рейтинг отелей города: {ra['avg_rating']}/10"
                f"\n- Доля положительных отзывов (8-10 баллов): {ra['positive_ratio']}%"
                f"\n- Что хвалят гости: {', '.join(ra['top_positive_words'][:7])}"
                f"\n- На что жалуются: {', '.join(ra['top_negative_words'][:7])}"
                f"\n- Отели с лучшими отзывами: {', '.join(ra['best_hotels'][:3])}"
            )
            
            if ra['sample_positive']:
                reviews_info += f"\n- Пример позитивного отзыва: \"{ra['sample_positive'][0][:150]}...\""
            if ra['sample_negative']:
                reviews_info += f"\n- Пример негативного отзыва: \"{ra['sample_negative'][0][:150]}...\""
        
        prompt = (
            f"Ты эксперт MTS Travel. Напиши роскошное описание для отеля {hotel_name} в городе {city_name}. "
            f"\n\n📍 ДОСТОПРИМЕЧАТЕЛЬНОСТИ РЯДОМ: {market_data['landmarks']}."
            f"\n\n🏨 АНАЛИЗ КОНКУРЕНТОВ ({market_data['count']} отелей): {market_data['raw_services_text']}."
            f"{reviews_info}"
            f"\n\n✨ НАШИ УНИКАЛЬНЫЕ ПРЕИМУЩЕСТВА: {features}."
            f"\n\n📝 ЗАДАЧА:"
            f"\n1. Проанализируй достопримечательности и включи их в текст"
            f"\n2. Учти, что хвалят гости в других отелях, и подчеркни, что у нас это тоже есть"
            f"\n3. Учти жалобы гостей и покажи, что у нас этих проблем нет"
            f"\n4. Выдели наши уникальные преимущества на фоне конкурентов"
            f"\n\n🎨 ФОРМАТ:"
            f"\n- Используй подходящие эмодзи"
            f"\n- Не используй символы # или * для заголовков"
            f"\n- Раздели текст на 3 красивых абзаца"
        )
        
        for attempt in range(3):
            try:
                with GigaChat(credentials=self.creds, verify_ssl_certs=False, timeout=30) as giga:
                    response = giga.chat(prompt)
                    text = response.choices[0].message.content
                    text = text.replace('#', '').replace('**', '')
                    return text
            except Exception as e:
                if attempt < 2:
                    time.sleep(3)
                    continue
                return f"Ошибка GigaChat: {str(e)}"

# --- Интерфейс ---
st.set_page_config(page_title="MTS Travel Аналитик", layout="wide")

GIGA_AUTH = "MDE5YjJlNGUtZTJiOC03Y2RiLWI0N2MtZDBjZDBkYzJmMDgxOmM1MjkyMzNlLTMxOTYtNDJlYi1hYjAzLTYwYzg4ODk0MGY5ZQ=="

if 'assistant' not in st.session_state:
    st.session_state.assistant = HotelAIAssistant(GIGA_AUTH)

assistant = st.session_state.assistant

with st.sidebar:
    st.header("🏨 Настройки отеля")
    
    h_name = st.text_input("Ваш отель", "Pino Select")
    
    city_options = sorted(list(assistant.city_map.keys()))
    c_name = st.selectbox("Выберите город для анализа", options=city_options)
    
    current_landmarks = assistant.city_map[c_name]["landmarks"]
    st.caption(f"📍 Будут учтены: {current_landmarks}")
    
    # Показываем статистику отзывов для выбранного города
    if assistant.reviews_df is not None:
        reviews_analysis, _ = assistant.get_reviews_analysis(c_name)
        if reviews_analysis:
            st.divider()
            st.subheader("📊 Статистика отзывов")
            st.metric("Всего отзывов", reviews_analysis['total_reviews'])
            st.metric("Средний рейтинг", f"{reviews_analysis['avg_rating']}/10")
            st.metric("👍 Положительных", f"{reviews_analysis['positive_ratio']}%")
            
            with st.expander("🔍 Детали анализа"):
                st.write("**Что хвалят:**")
                for w in reviews_analysis['top_positive_words'][:5]:
                    st.write(f"✅ {w}")
                st.write("**На что жалуются:**")
                for w in reviews_analysis['top_negative_words'][:5]:
                    st.write(f"❌ {w}")
    
    h_features = st.text_area("Ваши преимущества", 
                             "Панорамный вид, завтрак «шведский стол», SPA-комплекс, бассейн с подогревом, бесплатный Wi-Fi, круглосуточный сервис, фитнес-центр, трансфер, халат и тапочки, кофемашина в номере, звукоизоляция, меню подушек, консьерж-сервис, парковка под охраной, детская игровая зона.",
                             height=200)
    
    st.divider()
    
    st.subheader("📂 Состояние базы")
    st.metric("Всего отелей в системе", len(assistant.all_hotels_data))
    
    with st.expander(f"Загруженные файлы ({len(assistant.file_names)})"):
        for f in assistant.file_names:
            st.write(f"📄 {f}")
    
    if st.button("🔄 Обновить базу файлов", use_container_width=True):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.session_state.assistant = HotelAIAssistant(GIGA_AUTH)
        st.rerun()

# --- Главный экран ---
st.title("🏨 Генератор описаний на основе анализа рынка и отзывов")

if st.button("🚀 Запустить анализ и генерацию", use_container_width=True):
    with st.spinner("🔄 Изучаем конкурентов, отзывы и достопримечательности..."):
        market_data, error = assistant.get_market_context(c_name)
        
        if error and not market_data:
            st.error(error)
        else:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("📍 Сводка по рынку")
                st.info(f"Найдено конкурентов: **{market_data['count']}**")
                
                # Статистика отзывов
                if market_data.get('reviews_analysis'):
                    ra = market_data['reviews_analysis']
                    st.success(f"📊 Отзывов проанализировано: **{ra['total_reviews']}**")
                    st.write(f"⭐ Средний рейтинг: **{ra['avg_rating']}/10**")
                    st.write(f"👍 Доля позитивных: **{ra['positive_ratio']}%**")
                
                st.write("**Топ-15 частых услуг:**")
                for s, count in market_data['top_services'][:15]:
                    st.write(f"✅ {s} ({count})")
                
                with st.expander("🏨 Список отелей города"):
                    for name in market_data['names']:
                        st.write(f"• {name}")
                
                with st.expander("📝 Анализ отзывов (детали)"):
                    if market_data.get('reviews_analysis'):
                        ra = market_data['reviews_analysis']
                        st.write("**Лучшие отели по отзывам:**")
                        for h in ra['best_hotels'][:5]:
                            st.write(f"🏆 {h}")
                        st.write("**Худшие отели по отзывам:**")
                        for h in ra['worst_hotels'][:5]:
                            st.write(f"⚠️ {h}")
                        st.write("**Пример позитивного отзыва:**")
                        if ra['sample_positive']:
                            st.info(ra['sample_positive'][0][:200] + "...")
                        st.write("**Пример негативного отзыва:**")
                        if ra['sample_negative']:
                            st.warning(ra['sample_negative'][0][:200] + "...")
            
            with col2:
                st.subheader("📝 Сгенерированное описание")
                raw_result = assistant.generate_description(h_name, c_name, h_features, market_data)
                clean_result = raw_result.replace('\n', '<br>')
                
                st.markdown(
                    f"""
                    <div style="
                        border: 1px solid #464b5d; 
                        border-radius: 10px; 
                        padding: 25px; 
                        background-color: #262730; 
                        color: #ffffff;
                        line-height: 1.8;
                        font-size: 1.1rem;
                    ">
                        {clean_result}
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

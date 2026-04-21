import json
from bs4 import BeautifulSoup
import re

# Твоя функция остается без изменений, добавили только аргумент для имени выходного файла
def parse_multi_hotel_v3(filename, output_filename):
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    sections = re.split(r'(?i)<!DOCTYPE html>|<html', content)
    
    final_results = []
    sections = [s for s in sections if len(s) > 100] 

    print(f"Обработка файла {filename}: найдено блоков {len(sections)}")

    for section in sections:
        if len(section) < 500: continue 
        
        soup = BeautifulSoup('<html>' + section, 'lxml')

        name = "Без названия"
        h1 = soup.find('h1')
        if h1:
            name = h1.get_text(strip=True).split('—')[0].strip()
        else:
            title = soup.find('title')
            if title:
                name = title.get_text(strip=True).split('—')[0].split('*')[0].split(',')[0].strip()

        desc_meta = soup.find('meta', attrs={'name': 'description'})
        meta_text = desc_meta.get('content', '') if desc_meta else ""
        main_text_div = soup.find('div', class_='description_text')
        main_text = main_text_div.get_text(strip=True) if main_text_div else ""
        full_description = f"{meta_text} {main_text}".strip()

        services = []
        service_tags = soup.find_all(['span', 'div', 'li'], class_=re.compile(r'facility|service|amenity'))
        
        for tag in service_tags:
            text = tag.get_text(strip=True)
            if text and len(text) > 3 and text not in services:
                services.append(text)

        address = ""
        address_tag = soup.find('span', class_='hotel_address') or soup.find('div', class_='address')
        if address_tag:
            address = address_tag.get_text(strip=True)

        final_results.append({
            "name": name,
            "address": address,
            "description": full_description,
            "services_count": len(services),
            "services": services
        })

    # Теперь имя файла берется из настроек в конце кода
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)

    return len(final_results)

if __name__ == "__main__":
    # СПИСОК ФАЙЛОВ: "что читаем": "куда сохраняем"
    # Просто добавляй сюда новые строки по аналогии
    files_to_process = {
        "page_irkutsk.txt": "irkutsk_15_hotels.json",
        "page_voronezh.txt": "voronezh_15_hotels.json",
        "page_sochi.txt": "sochi_15_hotels.json",
        "page_saint_petersburg.txt": "saint_petersburg_15_hotels.json",
        "page_moscow.txt": "moscow_15_hotels.json",
        "page_kazan.txt": "kazan_15_hotels.json",
        "page_arkhyz.txt": "arkhyz_15_hotels.json",
        "page_krasnoyarsk.txt": "krasnoyarsk_15_hotels.json",
        "page_nizhniy_novgorod.txt": "nizhniy_novgorod_15_hotels.json",
        "page_novosibirsk.txt": "novosibirsk_15_hotels.json",
        "page_perm.txt": "perm_15_hotels.json",
        "page_petropavlovsk_kamchatsky.txt": "petropavlovsk_kamchatsky_15_hotels.json",
        "page_vladimir.txt": "vladimir_15_hotels.json",
        "page_vladivostok.txt": "vladivostok_15_hotels.json",
        "page_volgograd.txt": "volgograd_15_hotels.json"
    }

    for input_file, output_file in files_to_process.items():
        try:
            count = parse_multi_hotel_v3(input_file, output_file)
            print(f"Готово! {input_file} -> {output_file} (Отелей: {count})")
        except Exception as e:
            print(f"Ошибка при обработке {input_file}: {e}")

    print("\nПакетная обработка завершена!")

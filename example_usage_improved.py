#!/usr/bin/env python3
"""
Пример использования улучшенного детектора ИИ-текста
"""

from ai_detector_improved import AITextDetector
import json


def example_basic_usage():
    """Пример базового использования детектора"""
    print("=== Пример базового использования ===")
    
    # Создаем детектор
    detector = AITextDetector()
    
    # Пример текста для анализа (можно заменить на чтение из файла)
    sample_text = """
    В современном мире технологии играют важную роль. 
    В настоящее время наблюдается значительный рост использования цифровых технологий. 
    В условиях цифровизации общества важно обеспечивать доступ к информации. 
    В связи с вышеизложенным, следует отметить, что технологии развивают общество. 
    Более того, в контексте современных реалий, цифровизация влияет на все сферы жизни. 
    В заключение стоит отметить, что технологии являются важным элементом современного общества.
    """
    
    # Анализируем текст
    analysis = detector.analyze_text(sample_text)
    suggestions = detector.generate_suggestions(sample_text, analysis)
    
    print(f"Вероятность ИИ-текста: {analysis['ai_probability']:.1f}%")
    print(f"Найдено проблем: {len(analysis['issues'])}")
    print(f"Предложений по улучшению: {len(suggestions)}")
    
    # Применяем исправления
    fixed_text = detector.fix_text(sample_text)
    print(f"\nОригинальный текст:\n{sample_text[:200]}...")
    print(f"\nИсправленный текст:\n{fixed_text[:200]}...")


def example_file_processing():
    """Пример обработки DOCX файла"""
    print("\n=== Пример обработки файла ===")
    
    detector = AITextDetector()
    
    # Обработка файла (файл должен существовать)
    # Для примера используем try-except блок
    try:
        results = detector.process_docx_file('sample_ai_text.docx')
        
        if 'error' not in results:
            print(f"Файл: {results['file_path']}")
            print(f"Вероятность ИИ: {results['analysis']['ai_probability']:.1f}%")
            print(f"Проблем найдено: {len(results['analysis']['issues'])}")
            print(f"Предложений: {len(results['suggestions'])}")
            
            # Показать первые несколько предложений
            if results['suggestions']:
                print("\nПервые 3 предложения:")
                for i, suggestion in enumerate(results['suggestions'][:3], 1):
                    print(f"  {i}. {suggestion['issue']}")
        else:
            print(f"Ошибка при обработке файла: {results['error']}")
            print("Файл не найден, используем пример с текстом...")
            
            # Пример с созданием и анализом тестового файла
            test_doc = detector.fix_text("В современном мире технологии играют важную роль. В настоящее время наблюдается значительный рост.")
            print(f"Результат анализа тестового текста: {test_doc}")
    
    except Exception as e:
        print(f"Произошла ошибка: {e}")


def example_detailed_analysis():
    """Пример детального анализа"""
    print("\n=== Пример детального анализа ===")
    
    detector = AITextDetector()
    
    sample_text = """
    В современном мире наблюдается значительное влияние технологий на общество. 
    В настоящее время технологии развиваются с невероятной скоростью. 
    В условиях цифровизации важно обеспечивать доступ к информации. 
    В связи с вышеизложенным, необходимо учитывать все аспекты. 
    Более того, в контексте современных реалий, цифровизация влияет на все сферы жизни. 
    Кроме того, в рамках реализации программ, осуществляется внедрение новых решений. 
    Следовательно, в результате проведенных исследований, можно сделать вывод. 
    В заключение стоит отметить, что технологии являются важным элементом современного общества.
    """
    
    analysis = detector.analyze_text(sample_text)
    
    print("Детальная статистика:")
    print(f"  - Длина текста: {analysis['total_chars']} символов, {analysis['total_words']} слов")
    print(f"  - Количество предложений: {analysis['total_sentences']}")
    
    if analysis['sentence_stats']:
        stats = analysis['sentence_stats']
        print(f"  - Средняя длина предложения: {stats['avg_sentence_length']:.1f} слов")
        print(f"  - Доля сложных предложений: {stats['complex_sentence_ratio']*100:.1f}%")
        print(f"  - Вариативность длины предложений: {stats['sentence_variance']:.1f}")
    
    if analysis['word_stats']:
        stats = analysis['word_stats']
        print(f"  - Уникальные слова: {stats['uniqueness_ratio']*100:.1f}%")
        print(f"  - Средняя длина слова: {stats['avg_word_length']:.1f} символов")
        print(f"  - Доля формальных слов: {stats['formal_ratio']*100:.1f}%")
    
    print(f"  - Общая вероятность ИИ: {analysis['ai_probability']:.1f}%")


if __name__ == "__main__":
    example_basic_usage()
    example_file_processing()
    example_detailed_analysis()
    
    print("\n=== Дополнительная информация ===")
    print("Для полной работы с файлами используйте команды:")
    print("python ai_detector_improved.py document.docx")
    print("python ai_detector_improved.py document.docx --detailed")
    print("python ai_detector_improved.py document.docx --fix --fix-output fixed.docx")
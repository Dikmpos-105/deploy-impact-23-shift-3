# Generated by Django 4.2.5 on 2023-10-24 11:58

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0029_candidates_hard_skill_test_matching_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='associations',
            name='first_name',
            field=models.CharField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='associations',
            name='last_name',
            field=models.CharField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='associations',
            name='preferred_name',
            field=models.CharField(blank=True, null=True),
        ),
    ]